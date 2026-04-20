import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym
from stable_baselines3 import PPO
import sys
from scripts.lunar_lander_nav import LunarLander_Nav

def analyze_height_profile(model, n_eval_episodes=500, render_mode=None):
    env = gym.make("LunarLander_Nav", max_episode_steps=500, render_mode=render_mode)
    
    height_per_timestep = defaultdict(list)

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        step_count = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # obs[1] is the y position (height)
            height_per_timestep[step_count].append(obs[1])
            step_count += 1

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{n_eval_episodes} done")

    env.close()

    timesteps = sorted(height_per_timestep.keys())
    avg_height = [np.mean(height_per_timestep[t]) for t in timesteps]
    std_height  = [np.std(height_per_timestep[t])  for t in timesteps]
    alive       = [len(height_per_timestep[t])      for t in timesteps]

    # --- Plot 1: avg height per timestep ---
    fig, ax1 = plt.subplots(figsize=(12, 5))
    mean_std = np.mean(std_height)
    ax1.plot(timesteps, avg_height, label="Avg height", color="steelblue")
    ax1.fill_between(timesteps,
                     np.array(avg_height) - np.array(std_height),
                     np.array(avg_height) + np.array(std_height),
                     alpha=0.2, color="steelblue", label=f"Mean std: {mean_std:.2f}")
    
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Height (y)")
    ax1.set_title("Average Height per Timestep")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # --- Plot 2: episodes still alive (reliability) ---
    ax2 = ax1.twinx()
    ax2.plot(timesteps, alive, color="gray", linestyle="--", alpha=0.5, label="Episodes alive")
    ax2.set_ylabel("Episodes still alive", color="gray")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("height_profile.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\nPlot saved to height_profile.png")

    return timesteps, avg_height, alive


if __name__ == "__main__":
    model_str = sys.argv[1]
    model = PPO.load(model_str)
    print(f"Model loaded: {model_str}")

    analyze_height_profile(model, n_eval_episodes=500)