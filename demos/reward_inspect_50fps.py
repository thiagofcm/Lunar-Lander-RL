import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from scripts.lunar_lander_var_to_high import LunarLander_Var_to_High

from wrappers.common import TimeLimit


def summarize_episode(ep_idx, ep_data):
    nav = np.array(ep_data["nav_reward"])
    pen = np.array(ep_data["fps_penalty"])
    rew = np.array(ep_data["total_reward"])
    fps = np.array(ep_data["chosen_fps"])

    print(f"\nEpisode {ep_idx}")
    print("-" * 50)
    print(f"steps:                    {len(rew)}")
    print(f"episode total reward:     {rew.sum():.3f}")
    print(f"episode total nav reward: {nav.sum():.3f}")
    print(f"episode total fps pen:    {pen.sum():.3f}")
    print(f"mean total reward/step:   {rew.mean():.4f}")
    print(f"mean nav reward/step:     {nav.mean():.4f}")
    print(f"mean fps penalty/step:    {pen.mean():.4f}")
    print(f"mean chosen fps:          {fps.mean():.3f}")

    if len(rew) >= 50:
        print(f"last 50 mean nav reward:  {nav[-50:].mean():.4f}")
        print(f"last 50 mean fps:         {fps[-50:].mean():.3f}")


if __name__ == "__main__":
    model_str = sys.argv[1]

    eval_env = gym.make("LunarLander_Var_to_High")
    eval_env = TimeLimit(eval_env, max_episode_steps=500)

    model = PPO.load(model_str)
    print(f"Model loaded: {model_str}")

    n_eval_episodes = 10

    all_episode_returns = []
    all_episode_nav_returns = []
    all_episode_penalties = []
    all_episode_mean_fps = []

    all_episodes_data = []

    for ep in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        truncated = False

        ep_data = {
            "step": [],
            "nav_reward": [],
            "fps_penalty": [],
            "total_reward": [],
            "chosen_fps": [],
            "y": [],
            "vy": [],
        }

        step_idx = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)

            # --- Recover values ---
            fps_penalty = info.get("fps_penalty", 0.0)
            total_reward = info.get("reward", reward)
            nav_reward = total_reward + fps_penalty
            chosen_fps = info.get("chosen_fps", np.nan)

            # --- Store ---
            ep_data["step"].append(step_idx)
            ep_data["nav_reward"].append(nav_reward)
            ep_data["fps_penalty"].append(fps_penalty)
            ep_data["total_reward"].append(total_reward)
            ep_data["chosen_fps"].append(chosen_fps)

            # Extract state (last timestep if sequence)
            obs_arr = np.array(obs)

            if obs_arr.ndim == 2:
                state = obs_arr[-1, :8]
            else:
                state = obs_arr[:8]

            ep_data["y"].append(state[1])
            ep_data["vy"].append(state[3])

            step_idx += 1

        all_episodes_data.append(ep_data)

        ep_return = np.sum(ep_data["total_reward"])
        ep_nav_return = np.sum(ep_data["nav_reward"])
        ep_penalty = np.sum(ep_data["fps_penalty"])
        ep_mean_fps = np.nanmean(ep_data["chosen_fps"])

        all_episode_returns.append(ep_return)
        all_episode_nav_returns.append(ep_nav_return)
        all_episode_penalties.append(ep_penalty)
        all_episode_mean_fps.append(ep_mean_fps)

        summarize_episode(ep + 1, ep_data)

    eval_env.close()

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"mean total reward: {np.mean(all_episode_returns):.3f}")
    print(f"mean nav reward:   {np.mean(all_episode_nav_returns):.3f}")
    print(f"mean penalty:      {np.mean(all_episode_penalties):.3f}")
    print(f"mean chosen fps:   {np.mean(all_episode_mean_fps):.3f}")

    # =========================
    # Plot: step rewards
    # =========================
    first_ep = all_episodes_data[0]
    steps = first_ep["step"]

    plt.figure(figsize=(12, 5))
    plt.plot(steps, first_ep["nav_reward"], label="Nav Reward")
    plt.plot(steps, first_ep["fps_penalty"], label="FPS Penalty")
    plt.plot(steps, first_ep["total_reward"], label="Total Reward")
    plt.legend()
    plt.grid(True)
    plt.title("Per-step rewards")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.show()

    # =========================
    # Plot: FPS vs time
    # =========================
    plt.figure(figsize=(12, 5))
    plt.plot(steps, first_ep["chosen_fps"], label="Chosen FPS")
    plt.legend()
    plt.grid(True)
    plt.title("FPS over time")
    plt.xlabel("Step")
    plt.ylabel("FPS")
    plt.show()

    # =========================
    # Plot: state context
    # =========================
    plt.figure(figsize=(12, 5))
    plt.plot(steps, first_ep["y"], label="y")
    plt.plot(steps, first_ep["vy"], label="vy")
    plt.legend()
    plt.grid(True)
    plt.title("State (y, vy)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.show()