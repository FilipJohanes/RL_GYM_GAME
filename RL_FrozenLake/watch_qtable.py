import os
import time
import re
import numpy as np
import gymnasium as gym

ENV_ID = "FrozenLake-v1"
IS_SLIPPERY = False

def extract_episode_number(filename: str) -> int:
    m = re.search(r"ep(\d+)", filename)
    return int(m.group(1)) if m else -1

def list_checkpoints(q_dir: str):
    files = [f for f in os.listdir(q_dir) if f.endswith(".npy")]
    files.sort(key=extract_episode_number)
    return files

def play_one_episode(env, Q, step_delay=0.35, max_steps=200, verbose=True):
    # Basic sanity checks
    if Q.ndim != 2:
        raise ValueError(f"Q must be 2D, got shape {Q.shape}")
    if Q.shape[0] != env.observation_space.n or Q.shape[1] != env.action_space.n:
        raise ValueError(
            f"Q shape {Q.shape} does not match env "
            f"({env.observation_space.n}, {env.action_space.n})"
        )

    state, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0.0

    while not done and steps < max_steps:
        # Important: state must be int index
        s = int(state)
        action = int(np.argmax(Q[s]))

        # Print a little debug so you can see it's progressing
        if verbose:
            print(f" step {steps:03d} | state={s:02d} | action={action}")

        # Small delay so you can see movement
        time.sleep(step_delay)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        state = next_state
        steps += 1

    return total_reward, steps, done

def main(q_dir="q_tables", delay=0.35, pause=1.0, max_steps=200):
    files = list_checkpoints(q_dir)
    if not files:
        raise RuntimeError(f"No .npy checkpoints found in: {q_dir}")

    print("Found checkpoints:")
    for f in files:
        print(" ", f)

    # ✅ Keep ONE window open and reuse it (more reliable)
    env = gym.make(ENV_ID, is_slippery=IS_SLIPPERY, render_mode="human")

    try:
        for f in files:
            path = os.path.join(q_dir, f)
            ep = extract_episode_number(f)
            print(f"\n=== Playing {f} (episode {ep}) ===")

            Q = np.load(path)

            reward, steps, done = play_one_episode(
                env, Q, step_delay=delay, max_steps=max_steps, verbose=False
            )
            print(f"Result: reward={reward}, steps={steps}, done={done}")

            time.sleep(pause)

    finally:
        env.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="q_tables")
    p.add_argument("--delay", type=float, default=0.35)
    p.add_argument("--pause", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=200)
    args = p.parse_args()

    main(q_dir=args.dir, delay=args.delay, pause=args.pause, max_steps=args.max_steps)
