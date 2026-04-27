from random import random

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import scripts.lunar_lander_var_fps as lunar_lander_var_fps
import numpy as np
import sys

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

if __name__ == "__main__":

    model_str = sys.argv[1]

    # Start Evaluation
    eval_env = gym.make("LunarLander_VarFramerate", render_mode="human")
    eval_env = TimeLimit(eval_env, max_episode_steps=500)

    n_eval_episodes = 10
    episode_rewards = []
    episode_fps = []
    n_gif_episodes = 2
    gif_fps = 30

    #model_str = "lunar_lander_models\\navigation\\30-03-2026_21-49-49\\ppo-nav.zip"
    model = PPO.load(model_str)
    print(f"Model Loaded: {model_str}")

    for ep in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        total_reward = 0
        touchdown_flag = False

        unwrapped = eval_env.unwrapped
        W = 600 / 30.0
        helipad_x1_norm = (unwrapped.helipad_x1 - W/2) / (W/2)
        helipad_x2_norm = (unwrapped.helipad_x2 - W/2) / (W/2)

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
            episode_fps.append(info["chosen_fps"])
            print(f"Chosen FPS: {info['chosen_fps']}")
            print("===============================")
            # # Print x, y at first touchdown
            # if (obs[-1][6] or obs[-1][7]) and not touchdown_flag:
            #             print(f"Episode {ep+1} | Touchdown → x: {obs[-1][0]:.4f} | Flags: [{helipad_x1_norm:.4f}, {helipad_x2_norm:.4f}]")
            #             landed_in_flags = helipad_x1_norm < obs[-1][0] < helipad_x2_norm
            #             print(f"Landed in flags: {landed_in_flags}")
            #             touchdown_flag = True

        episode_rewards.append(total_reward)

    eval_env.close()

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
    plt.show()