import numpy as np
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.PettingZooWrapper import PettingZooWrapper
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


class ContinuousMAEnv:
    def __init__(self, env):
        self.env = env
        obs, _ = env.reset(seed=0)
        self.agent_ids = sorted(env.agents)
        self.n_agents = len(self.agent_ids)
        space = env.action_space(self.agent_ids[0])
        self.act_shape = space.shape
        self.act_dim = int(np.prod(space.shape))
        self.low = space.low.ravel().astype(np.float32)
        self.high = space.high.ravel().astype(np.float32)
        self.obs_dim = self._stack(obs).shape[1]
        self.ctrl_freq = 30

    def _stack(self, obs):
        return np.stack([np.asarray(obs[a], dtype=np.float32).ravel()
                         for a in self.agent_ids])

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        return self._stack(obs)

    def step(self, actions):
        clipped = np.clip(actions, self.low, self.high)
        act = {a: clipped[i].reshape(self.act_shape)
               for i, a in enumerate(self.agent_ids)}
        obs, rew, term, trunc, _ = self.env.step(act)
        rewards = np.array([rew[a] for a in self.agent_ids], dtype=np.float32)
        terminated = any(term[a] for a in self.agent_ids)
        truncated = any(trunc[a] for a in self.agent_ids)
        done = terminated or truncated or not self.env.agents
        return self._stack(obs), rewards, terminated, done

    def close(self):
        self.env.close()


def make_env(num_drones, act_type="ONE_D_RPM", gui=False):
    raw = MultiHoverAviary(num_drones=num_drones,
                           obs=ObservationType.KIN,
                           act=ActionType[act_type],
                           gui=gui)
    env = ContinuousMAEnv(PettingZooWrapper(raw))
    env.ctrl_freq = getattr(raw, "CTRL_FREQ", 30)
    return env