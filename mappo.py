import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):
        bm, bv, n = x.mean(0), x.var(0), x.shape[0]
        delta, tot = bm - self.mean, self.count + n
        self.mean = self.mean + delta * n / tot
        m2 = self.var * self.count + bv * n + delta ** 2 * self.count * n / tot
        self.var, self.count = m2 / tot, tot

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state(self):
        return {"mean": torch.as_tensor(np.asarray(self.mean)),
                "var": torch.as_tensor(np.asarray(self.var)),
                "count": float(self.count)}

    def load(self, d):
        self.mean = d["mean"].numpy().astype(np.float64)
        self.var = d["var"].numpy().astype(np.float64)
        self.count = float(d["count"])


def mlp(in_dim, out_dim, hidden=128, out_gain=1.0):
    net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                        nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, out_dim))
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, gain=out_gain)
    return net


class Actor(nn.Module):
    def __init__(self, in_dim, act_dim, init_log_std=-0.5):
        super().__init__()
        self.mu = mlp(in_dim, act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, x):
        return torch.distributions.Normal(self.mu(x), self.log_std.exp())


class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.v = mlp(in_dim, 1, out_gain=1.0)

    def forward(self, x):
        return self.v(x)


def actor_inputs(obs):
    B, n, _ = obs.shape
    ids = torch.eye(n).unsqueeze(0).expand(B, -1, -1)
    return torch.cat([obs, ids], -1)


def critic_inputs(obs):
    B, n, D = obs.shape
    state = obs.reshape(B, 1, n * D).expand(B, n, -1)
    ids = torch.eye(n).unsqueeze(0).expand(B, -1, -1)
    return torch.cat([state, ids], -1)


class RolloutBuffer:
    def __init__(self, T, n, obs_dim, act_dim):
        self.obs = np.zeros((T, n, obs_dim), np.float32)
        self.act = np.zeros((T, n, act_dim), np.float32)
        self.logp = np.zeros((T, n), np.float32)
        self.val = np.zeros((T, n), np.float32)
        self.rew = np.zeros((T, n), np.float32)
        self.end = np.zeros(T, np.float32)
        self.t = 0

    def add(self, obs, act, logp, val, rew, end):
        i = self.t
        self.obs[i], self.act[i], self.logp[i] = obs, act, logp
        self.val[i], self.rew[i], self.end[i] = val, rew, end
        self.t += 1


class MAPPO:
    def __init__(self, n, obs_dim, act_dim, lr=5e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=10, n_minibatch=4, ent_coef=0.0,
                 vf_coef=1.0, max_grad=10.0):
        self.n, self.obs_dim, self.act_dim = n, obs_dim, act_dim
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.n_minibatch = epochs, n_minibatch
        self.ent_coef, self.vf_coef, self.max_grad = ent_coef, vf_coef, max_grad
        self.actor = Actor(obs_dim + n, act_dim)
        self.critic = Critic(n * obs_dim + n)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)
        self.ret_rms = RunningMeanStd(())

    @torch.no_grad()
    def act(self, obs_n, deterministic=False):
        o = torch.as_tensor(obs_n).unsqueeze(0)
        d = self.actor.dist(actor_inputs(o)[0])
        a = d.mean if deterministic else d.sample()
        logp = d.log_prob(a).sum(-1)
        return a.numpy(), logp.numpy(), self.value(obs_n)

    @torch.no_grad()
    def value(self, obs_n):
        o = torch.as_tensor(obs_n).unsqueeze(0)
        v = self.critic(critic_inputs(o))[0, :, 0].numpy()
        return v * np.sqrt(self.ret_rms.var) + self.ret_rms.mean

    def _gae(self, buf, last_value):
        T = buf.t
        adv = np.zeros((T, self.n), np.float32)
        last = np.zeros(self.n, np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - buf.end[t]
            next_v = last_value if t == T - 1 else buf.val[t + 1]
            delta = buf.rew[t] + self.gamma * next_v * nonterm - buf.val[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        return adv, adv + buf.val[:T]

    def update(self, buf, last_value):
        T, n = buf.t, self.n
        adv, ret = self._gae(buf, last_value)

        self.ret_rms.update(ret.reshape(-1))
        ret_n = (ret - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + 1e-8)

        obs = torch.as_tensor(buf.obs[:T])
        a_in = actor_inputs(obs).reshape(T * n, -1)
        c_in = critic_inputs(obs).reshape(T * n, -1)
        act = torch.as_tensor(buf.act[:T]).reshape(T * n, -1)
        old_logp = torch.as_tensor(buf.logp[:T]).reshape(-1)
        adv = torch.as_tensor(adv).reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret_n = torch.as_tensor(ret_n, dtype=torch.float32).reshape(-1)

        N, mb = T * n, (T * n) // self.n_minibatch
        stats = dict(pi=0., v=0., kl=0., clipfrac=0., k=0)
        for _ in range(self.epochs):
            perm = torch.randperm(N)
            for k in range(self.n_minibatch):
                idx = perm[k * mb:(k + 1) * mb]
                d = self.actor.dist(a_in[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = (logp - old_logp[idx]).exp()
                s1 = ratio * adv[idx]
                s2 = ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]
                pi_loss = -torch.min(s1, s2).mean() - self.ent_coef * d.entropy().sum(-1).mean()
                v_loss = F.smooth_l1_loss(self.critic(c_in[idx]).squeeze(-1), ret_n[idx])

                self.opt_a.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
                self.opt_a.step()

                self.opt_c.zero_grad()
                (self.vf_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
                self.opt_c.step()

                with torch.no_grad():
                    stats["pi"] += pi_loss.item()
                    stats["v"] += v_loss.item()
                    stats["kl"] += (old_logp[idx] - logp).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > self.clip).float().mean().item()
                    stats["k"] += 1
        k = stats.pop("k")
        return {key: val / k for key, val in stats.items()}

    def save(self, path, obs_rms, act_type):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "obs_rms": obs_rms.state(),
            "ret_rms": self.ret_rms.state(),
            "meta": {"n": self.n, "obs_dim": self.obs_dim,
                     "act_dim": self.act_dim, "act_type": act_type},
        }, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path)
        meta = ckpt["meta"]
        algo = cls(meta["n"], meta["obs_dim"], meta["act_dim"])
        algo.actor.load_state_dict(ckpt["actor"])
        algo.critic.load_state_dict(ckpt["critic"])
        algo.ret_rms.load(ckpt["ret_rms"])
        obs_rms = RunningMeanStd((meta["obs_dim"],))
        obs_rms.load(ckpt["obs_rms"])
        return algo, obs_rms, meta