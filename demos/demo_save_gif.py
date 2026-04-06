import os
import sys
import imageio
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np


def smooth(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.timesteps = []
        self.rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.timesteps.append(self.num_timesteps)
                self.rewards.append(info["episode"]["r"])
        return True


if __name__ == "__main__":
    model_str = sys.argv[1]

    output_dir = f"demos/navigation/gif_outputs_criteria_{os.path.basename(model_str).split('.zip')[0]}"
    os.makedirs(output_dir, exist_ok=True)

    # Start Evaluation
    eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")

    n_eval_episodes = 500
    n_gif_episodes = 2
    gif_fps = 50
    episode_rewards = []

    model = PPO.load(model_str)
    print(f"Model Loaded: {model_str}")

    for ep in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        frames = []

        # Capture the initial frame only for the first 2 episodes
        if ep < n_gif_episodes:
            frame = eval_env.render()
            if frame is not None:
                frames.append(frame)

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward

            if ep < n_gif_episodes:
                frame = eval_env.render()
                if frame is not None:
                    frames.append(frame)

        episode_rewards.append(total_reward)
        print(f"Evaluation Episode {ep + 1}/{n_eval_episodes} | Reward: {total_reward:.2f}")

        # Save GIF for first 2 episodes
        if ep < n_gif_episodes and len(frames) > 0:
            gif_path = os.path.join(output_dir, f"eval_episode_{ep + 1}.gif")
            imageio.mimsave(gif_path, frames, fps=gif_fps)
            print(f"Saved GIF: {gif_path}")

    eval_env.close()

    mean_reward = np.mean(episode_rewards)
    print(f"mean_reward={mean_reward:.2f}")

    txt_path = os.path.join(output_dir, "evaluation_results.txt")

with open(txt_path, "w") as f:
    f.write(f"Model: {model_str}\n")
    f.write(f"Number of evaluation episodes: {n_eval_episodes}\n")
    f.write(f"Mean reward: {mean_reward:.4f}\n\n")

    # f.write("Episode Rewards:\n")
    # for i, r in enumerate(episode_rewards):
    #     f.write(f"Episode {i+1}: {r:.4f}\n")

print(f"Saved evaluation results to: {txt_path}")