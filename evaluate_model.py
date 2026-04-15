# =========================
# Imports
# =========================
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from gymnasium.wrappers import TimeLimit

# Your custom env
import scripts.lunar_lander_var_fps as lunar_lander_var_fps
from scripts.lunar_lander_var_fps import navigation_model_path
from scripts.lunar_lander_var_fps import FPS_COST


# =========================
# USER INPUT (ONLY CHANGE THESE)
# =========================
MODEL_PATH = "lunar_lander_models\\navigation\\15-04-2026_17-00-41\\ppo-nav.zip"
OUTPUT_DIR = "lunar_lander_models\\navigation\\15-04-2026_17-00-41\\plots_eval"


# =========================
# Utils
# =========================
def smooth(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


# =========================
# Load Model
# =========================
def load_model(path):
    try:
        model = RecurrentPPO.load(path)
        print("Loaded RecurrentPPO model")
    except:
        model = PPO.load(path)
        print("Loaded PPO model")
    return model


# =========================
# Main Evaluation
# =========================
def evaluate():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model(MODEL_PATH)

    # Create environment
    eval_env = gym.make("LunarLander_VarFramerate")
    eval_env = TimeLimit(eval_env, max_episode_steps=500)

    n_eval_episodes = 100

    episode_rewards = []
    mean_chosen_fps_eval = []
    fps_traces_all = []

    # =========================
    # Run Evaluation
    # =========================
    for ep in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        fps_trace = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

            total_reward += reward
            fps_trace.append(info["chosen_fps"])

        episode_rewards.append(total_reward)
        mean_chosen_fps_eval.append(np.mean(fps_trace))
        fps_traces_all.append(fps_trace)

        print(f"[Eval] Episode {ep+1}/{n_eval_episodes} | Reward: {total_reward:.2f}")

    eval_env.close()

    # =========================
    # Summary
    # =========================
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print("\n===== Evaluation Summary =====")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean FPS: {np.mean(mean_chosen_fps_eval):.2f}")

    # =========================
    # Plot: Reward vs Episode
    # =========================
    ep_eval = np.arange(1, len(episode_rewards) + 1)
    rew_eval = np.array(episode_rewards)

    rew_eval_s = smooth(rew_eval, window=20)
    ep_eval_s = ep_eval[len(ep_eval) - len(rew_eval_s):]

    plt.figure()
    plt.plot(ep_eval, rew_eval, alpha=0.3, label="Raw")
    plt.plot(ep_eval_s, rew_eval_s, linewidth=2, label="Smoothed")
    plt.ylim(-400, 350)
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Total Reward")
    plt.title("Evaluation Reward vs Episode")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_total_rew_vs_ep.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # Plot: Mean FPS vs Episode
    # =========================
    ep_fps_eval = np.arange(1, len(mean_chosen_fps_eval) + 1)
    mean_fps_eval = np.array(mean_chosen_fps_eval)

    mean_fps_s_eval = smooth(mean_fps_eval, window=20)
    ep_fps_s_eval = ep_fps_eval[len(ep_fps_eval) - len(mean_fps_s_eval):]

    plt.figure()
    plt.plot(ep_fps_eval, mean_fps_eval, alpha=0.3, label="Raw")
    plt.plot(ep_fps_s_eval, mean_fps_s_eval, linewidth=2, label="Smoothed")
    plt.ylim(0, 55)
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Mean Chosen FPS")
    plt.title("Evaluation Mean Chosen FPS vs Episode")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_mean_chosen_fps_vs_ep.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # Plot: FPS traces (3 random episodes)
    # =========================
    random_episode_indices = random.sample(range(n_eval_episodes), 3)

    for idx in random_episode_indices:
        fps_trace = fps_traces_all[idx]
        timesteps = np.arange(1, len(fps_trace) + 1)

        plt.figure()
        plt.plot(timesteps, fps_trace, linewidth=2)
        plt.xlabel("Timestep")
        plt.ylabel("Chosen FPS")
        plt.ylim(0, 55)
        plt.title(f"Chosen FPS vs Timestep - Episode {idx+1}")
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"chosen_fps_vs_timestep_ep_{idx+1}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    print(f"\nEvaluation plots saved to: {OUTPUT_DIR}")


# =========================
# Run
# =========================
if __name__ == "__main__":
    evaluate()

# python -m demos.demo_var_framerate lunar_lander_models\var_framerate\11-04-2026_17-35-15_FPS_0_0\ppo_var_fps_cost_0_0.zip