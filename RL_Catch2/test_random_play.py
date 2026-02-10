import time
import numpy as np
from tag_iso_env import TagIsoEnv

def main():
    env = TagIsoEnv(render_mode="human")
    obs, info = env.reset()
    done = False

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(0.01)
        if done:
            obs, info = env.reset()

if __name__ == "__main__":
    main()
