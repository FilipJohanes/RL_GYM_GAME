import os
import numpy as np
import gymnasium as gym

# -----------------------
# Config
# -----------------------
ENV_ID = "FrozenLake-v1"
IS_SLIPPERY = False

EPISODES = 2000
SAVE_EVERY = 100

ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPS_DECAY = 0.995
EPS_MIN = 0.05

OUT_DIR = "q_tables"

# -----------------------
# Setup
# -----------------------
os.makedirs(OUT_DIR, exist_ok=True)

env = gym.make(ENV_ID, is_slippery=IS_SLIPPERY)

n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions), dtype=np.float32)

epsilon = EPSILON_START

def evaluate_greedy(Q_table, trials=200) -> float:
    """Evaluate current policy with no exploration."""
    eval_env = gym.make(ENV_ID, is_slippery=IS_SLIPPERY)
    wins = 0
    for _ in range(trials):
        s, _ = eval_env.reset()
        done = False
        while not done:
            a = int(np.argmax(Q_table[s]))
            s, r, terminated, truncated, _ = eval_env.step(a)
            done = terminated or truncated
            if done and r == 1:
                wins += 1
    eval_env.close()
    return wins / trials

# -----------------------
# Training loop
# -----------------------
for ep in range(1, EPISODES + 1):
    state, _ = env.reset()
    done = False

    while not done:
        # epsilon-greedy action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        Q[state, action] += ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[state, action])
        state = next_state

    # decay epsilon after each episode
    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

    # save checkpoints
    if ep % SAVE_EVERY == 0:
        win_rate = evaluate_greedy(Q, trials=200)
        ckpt_path = os.path.join(OUT_DIR, f"Q_ep{ep:04d}.npy")
        np.save(ckpt_path, Q)
        print(f"Episode {ep:4d} | epsilon={epsilon:.3f} | greedy win_rate={win_rate:.2%} | saved {ckpt_path}")

env.close()
print("Training complete.")
