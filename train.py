import argparse
import time
import numpy as np
import torch

from env import make_env
from mappo import MAPPO, RolloutBuffer, RunningMeanStd


def evaluate(env, algo, obs_rms, episodes=3, seed=123):
    rets = []
    for ep in range(episodes):
        obs, done, total = env.reset(seed=seed + ep), False, np.zeros(env.n_agents)
        while not done:
            a, _, _ = algo.act(obs_rms.normalize(obs), deterministic=True)
            obs, r, _, done = env.step(a)
            total += r
        rets.append(total.mean())
    return float(np.mean(rets))


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(args.drones, args.act_type)
    n, D, A = env.n_agents, env.obs_dim, env.act_dim
    algo = MAPPO(n, D, A)
    obs_rms = RunningMeanStd((D,))

    obs = env.reset(seed=args.seed)
    obs_rms.update(obs)
    ep_ret, recent = np.zeros(n), []
    steps, updates, t0 = 0, 0, time.time()

    while steps < args.total_steps:
        buf = RolloutBuffer(args.n_steps, n, D, A)
        for _ in range(args.n_steps):
            o_n = obs_rms.normalize(obs)
            a, logp, v = algo.act(o_n)
            next_obs, r, terminated, done = env.step(a)
            ep_ret += r

            r_store = r.copy()
            if done and not terminated:
                r_store += algo.gamma * algo.value(obs_rms.normalize(next_obs))
            buf.add(o_n, a, logp, v, r_store, float(done))

            if done:
                recent.append(ep_ret.mean())
                ep_ret = np.zeros(n)
                obs = env.reset()
            else:
                obs = next_obs
            obs_rms.update(obs)
        steps += args.n_steps

        last_v = algo.value(obs_rms.normalize(obs))
        info = algo.update(buf, last_v)
        updates += 1

        mean_ret = np.mean(recent[-10:]) if recent else float("nan")
        print(f"steps {steps:8d} | ret {mean_ret:8.2f} "
              f"| std {algo.actor.log_std.exp().mean().item():.3f} "
              f"| kl {info['kl']:+.4f} | clipfrac {info['clipfrac']:.2f} "
              f"| v_loss {info['v']:.4f} | {steps / (time.time() - t0):.0f} sps")

        if updates % args.save_every == 0:
            algo.save(args.save, obs_rms, args.act_type)

    algo.save(args.save, obs_rms, args.act_type)
    print(f"final eval return: {evaluate(env, algo, obs_rms):.2f}")
    print(f"saved to {args.save}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--drones", type=int, default=3)
    p.add_argument("--act-type", default="ONE_D_RPM")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="mappo_checkpoint.pt")
    p.add_argument("--save-every", type=int, default=10)
    train(p.parse_args())