from random import random

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import scripts.lunar_lander_var_fps as lunar_lander_var_fps
import numpy as np
import sys
import cv2
import imageio

from wrappers.common import TimeLimit

def smooth(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="valid")

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

def add_fps_overlay(frame, chosen_fps, step):
    """Add FPS text overlay to a frame using OpenCV."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # FPS text
    cv2.putText(
        frame_bgr,
        f"FPS: {chosen_fps}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Step text
    cv2.putText(
        frame_bgr,
        f"Step: {step}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def save_gif(frames, path, gif_fps=30):
    imageio.mimsave(path, frames, fps=gif_fps)
    print(f"GIF saved to: {path}")

if __name__ == "__main__":

    model_str = sys.argv[1]
    model = PPO.load(model_str)
    print(f"Model Loaded: {model_str}")

    # Use rgb_array for GIF capture, human for display
    n_eval_episodes = 10
    n_gif_episodes = 2
    gif_fps = 30

    episode_rewards = []
    episode_fps = []

    for ep in range(n_eval_episodes):

        # Use rgb_array for gif episodes, human for the rest
        render_mode = "rgb_array" if ep < n_gif_episodes else "human"
        eval_env = gym.make("LunarLander_VarFramerate", render_mode=render_mode)
        eval_env = TimeLimit(eval_env, max_episode_steps=500)

        obs, _ = eval_env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        frames = []
        chosen_fps = 10  # default before first action

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
            chosen_fps = info["chosen_fps"]
            episode_fps.append(chosen_fps)

            # Capture frame for GIF episodes
            if ep < n_gif_episodes:
                frame = eval_env.render()
                if frame is not None:
                    frame_with_overlay = add_fps_overlay(frame, chosen_fps, step)
                    frames.append(frame_with_overlay)

            print(f"Ep {ep+1} | Step {step} | Chosen FPS: {chosen_fps}")
            step += 1

        episode_rewards.append(total_reward)
        eval_env.close()

        # Save GIF for first two episodes
        if ep < n_gif_episodes:
            gif_path = f"episode_{ep+1}_fps_demo.gif"
            save_gif(frames, gif_path, gif_fps=gif_fps)

        print(f"Episode {ep+1} | Total Reward: {total_reward:.2f}")
        print("===============================")

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"mean_reward={mean_reward:.2f}")

    # Plot Evaluation Results
    window_eval = 5
    smoothed_eval = smooth(episode_rewards, window_eval)
    smoothed_episodes = list(range(window_eval, len(episode_rewards)+1))

    plt.figure()
    plt.plot(episode_rewards, alpha=0.3, label="Raw Reward")
    plt.plot(smoothed_episodes, smoothed_eval, linewidth=2, label=f"Smoothed (window={window_eval})")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Episode Reward")
    plt.title("Evaluation Rewards (LunarLander)")
    plt.grid(True)
    plt.legend()
    plt.show()