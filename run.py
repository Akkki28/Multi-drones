import argparse
import time
import numpy as np

from env import make_env
from mappo import MAPPO


def run(args):
    algo, obs_rms, meta = MAPPO.load(args.checkpoint)
    env = make_env(
        meta["n"],
        meta["act_type"],
        gui=not args.no_gui,
        record=args.record_video,
        video_size=(args.video_width, args.video_height) if args.record_video else None,
    )

    returns = []
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        total, steps, done = np.zeros(env.n_agents), 0, False
        while not done:
            a, _, _ = algo.act(obs_rms.normalize(obs), deterministic=not args.stochastic)
            obs, r, _, done = env.step(a)
            total += r
            steps += 1
            if not args.no_gui:
                time.sleep(1.0 / env.ctrl_freq)
        returns.append(total.mean())
        print(f"episode {ep:3d} | steps {steps:4d} | return {total.mean():8.2f}")

    print(f"mean return over {args.episodes} episodes: {np.mean(returns):.2f}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="mappo_checkpoint.pt")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--record-video", action="store_true",
                   help="save the GUI run as an MP4 in the results directory")
    p.add_argument("--video-width", type=int, default=1920,
                   help="recorded video width in pixels (default: 1920)")
    p.add_argument("--video-height", type=int, default=1080,
                   help="recorded video height in pixels (default: 1080)")
    run(p.parse_args())