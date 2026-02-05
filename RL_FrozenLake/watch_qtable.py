import os
import time
import numpy as np
import gymnasium as gym
import re

ENV_ID = "FrozenLake-v1"
IS_SLIPPERY = False


def extract_episode_number(filename: str) -> int:
    """
    Extract episode number from filenames like:
    Q_ep0100.npy -> 100
    """
    match = re.search(r"ep(\d+)", filename)
    return int(match.group(1)) if match else -1


def watch_qtable(Q, delay=0.4):
    env = gym.make(
        ENV_ID,
        is_slippery=IS_SLIPPERY,
        render_mode="human"
    )

    state, _ = env.reset()
    done = False

    while not done:
        time.sleep(delay)
        action = int(np.argmax(Q[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    return reward


def main(q_dir: str, delay: float, pause_between: float):
    files = [
        f for f in os.listdir(q_dir)
        if f.endswith(".npy")
    ]

    if not files:
        raise RuntimeError(f"No .npy files found in {q_dir}")

    # sort by episode number
    files.sort(key=extract_episode_number)

    print("Found Q-tables:")
    for f in files:
        print(" ", f)

    for f in files:
        path = os.path.join(q_dir, f)
        ep = extract_episode_number(f)

        print(f"\n=== Watching checkpoint: {f} (episode {ep}) ===")
        Q = np.load(path)

        reward = watch_qtable(Q, delay=delay)
        print("Reward:", reward)

        time.sleep(pause_between)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch FrozenLake Q-learning checkpoints")
    parser.add_argument(
        "--dir",
        default="q_tables",
        help="Directory containing saved Q-table .npy files"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between steps (seconds)"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.5,
        help="Pause between checkpoints (seconds)"
    )

    args = parser.parse_args()

    main(
        q_dir=args.dir,
        delay=args.delay,
        pause_between=args.pause
    )
