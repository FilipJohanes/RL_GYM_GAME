from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from catch_env import CatchEnv

def main():
    # Vectorized env speeds up training
    env = make_vec_env(lambda: CatchEnv(render_mode=None), n_envs=8)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
    )

    # Train
    model.learn(total_timesteps=200_000)

    # Save
    model.save("ppo_catch")
    env.close()
    print("Saved model to ppo_catch.zip")

if __name__ == "__main__":
    main()
