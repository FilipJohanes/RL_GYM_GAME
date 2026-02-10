import time
from stable_baselines3 import PPO
from catch_env import CatchEnv

def main():
    env = CatchEnv(render_mode="human")
    model = PPO.load("ppo_catch")

    episodes = 20
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total += reward
            time.sleep(0.02)

        print(f"Episode {ep+1}: total_reward={total}")

    env.close()

if __name__ == "__main__":
    main()
