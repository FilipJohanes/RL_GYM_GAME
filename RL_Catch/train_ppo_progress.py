import os
import time
import numpy as np

from stable_baselines3 import PPO
from catch_env import CatchEnv


def evaluate_catch_rate(model: PPO, n_episodes: int = 200) -> float:
    """Returns catch rate in [0, 1]. Reward is +1 for catch, -1 for miss."""
    env = CatchEnv(render_mode=None)
    catches = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            ep_reward += reward

        if ep_reward > 0:  # +1 means caught
            catches += 1

    env.close()
    return catches / n_episodes


def demo_render(model: PPO, n_episodes: int = 5, step_delay: float = 0.02):
    """Render a few episodes so you can SEE the policy."""
    env = CatchEnv(render_mode="human")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total += reward
            time.sleep(step_delay)
        print(f"[DEMO] Episode {ep+1}/{n_episodes} total_reward={total}")
    env.close()


def main():
    os.makedirs("checkpoints", exist_ok=True)

    # Train env (no rendering, faster)
    train_env = CatchEnv(render_mode=None)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
    )

    # Progress gating parameters
    eval_episodes = 200
    improvement_threshold = 0.05  # 5 percentage points
    train_chunk = 10_000          # train this many timesteps per loop
    demo_episodes = 5

    best_shown_rate = 0.0
    total_trained = 0

    # Initial evaluation (untrained)
    rate = evaluate_catch_rate(model, n_episodes=eval_episodes)
    print(f"Initial catch rate: {rate:.2%}")
    best_shown_rate = rate
    demo_render(model, n_episodes=demo_episodes)

    while True:
        # Train a bit more
        model.learn(total_timesteps=train_chunk, reset_num_timesteps=False)
        total_trained += train_chunk

        # Evaluate
        rate = evaluate_catch_rate(model, n_episodes=eval_episodes)
        print(f"[EVAL] timesteps={total_trained:,} catch_rate={rate:.2%} (last shown {best_shown_rate:.2%})")

        # Show progress only if improved enough
        if rate >= best_shown_rate + improvement_threshold:
            best_shown_rate = rate
            ckpt_path = f"checkpoints/ppo_catch_{total_trained:07d}_rate_{int(rate*100):02d}.zip"
            model.save(ckpt_path)
            print(f"[CHECKPOINT] Saved: {ckpt_path}")
            demo_render(model, n_episodes=demo_episodes)

    train_env.close()


if __name__ == "__main__":
    main()
