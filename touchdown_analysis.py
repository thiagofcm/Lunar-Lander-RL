import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
import sys

def analyze_touchdown_velocities(model, n_eval_episodes=500, render_mode=None):
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    touchdown_data = []

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        landed = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            if (obs[6] or obs[7]) and not landed:
                touchdown_data.append({
                    "vy": abs(obs[3]),
                    "vx": abs(obs[2]),
                    "tilt": abs(obs[4]),
                })
                landed = True

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{n_eval_episodes} done")

    env.close()

    df = pd.DataFrame(touchdown_data)

    print("\n===== Touchdown Velocity Statistics =====")
    print(df.describe())

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df["vy"], bins=30, color="steelblue", edgecolor="black")
    axes[0].set_title("Vertical Velocity at Touchdown")
    axes[0].set_xlabel("|vy|")
    axes[0].set_ylabel("Count")
    axes[0].axvline(df["vy"].mean(), color="red", linestyle="--", label=f"mean={df['vy'].mean():.2f}")
    axes[0].legend()

    axes[1].hist(df["vx"], bins=30, color="seagreen", edgecolor="black")
    axes[1].set_title("Horizontal Velocity at Touchdown")
    axes[1].set_xlabel("|vx|")
    axes[1].set_ylabel("Count")
    axes[1].axvline(df["vx"].mean(), color="red", linestyle="--", label=f"mean={df['vx'].mean():.2f}")
    axes[1].legend()

    axes[2].hist(df["tilt"], bins=30, color="darkorange", edgecolor="black")
    axes[2].set_title("Tilt Angle at Touchdown")
    axes[2].set_xlabel("|tilt|")
    axes[2].set_ylabel("Count")
    axes[2].axvline(df["tilt"].mean(), color="red", linestyle="--", label=f"mean={df['tilt'].mean():.2f}")
    axes[2].legend()

    plt.suptitle("Touchdown Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig("touchdown_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\nPlot saved to touchdown_analysis.png")

    return df


if __name__ == "__main__":
    model_str = sys.argv[1]
    model = PPO.load(model_str)
    print(f"Model loaded: {model_str}")

    df = analyze_touchdown_velocities(model, n_eval_episodes=500)