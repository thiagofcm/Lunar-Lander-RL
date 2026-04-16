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

import scripts.lunar_lander_var_fps as lunar_lander_var_fps
from scripts.lunar_lander_var_fps import navigation_model_path

# =========================
# USER INPUT (ONLY CHANGE THESE)
# =========================
VAR_MODEL_PATH   = "lunar_lander_models\\var_framerate_per_frame_penalty\\16-04-2026_09-01-55_FPS_0_6\\plots\\checkpoint_ep_16000\\model.zip"
NAV_MODEL_PATH   = "lunar_lander_models\\navigation\\15-04-2026_17-00-41\\ppo-nav.zip"
OUTPUT_DIR       = "lunar_lander_models\\var_framerate_per_frame_penalty\\16-04-2026_09-01-55_FPS_0_6\\plots\\checkpoint_ep_16000\\comparison_eval"
FRAME_COST       = 0.6
N_EVAL_EPISODES  = 100
N_RUNS           = 10
FIXED_FPS_LIST   = [1,5,10,25]  # baselines to compare against
fps_to_action = {1: 0, 5: 1, 10: 2, 25: 3}  # maps FPS to action index

# =========================
# Utils
# =========================
def smooth(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def load_model(path):
    try:
        model = RecurrentPPO.load(path)
        print(f"Loaded RecurrentPPO: {path}")
    except:
        model = PPO.load(path)
        print(f"Loaded PPO: {path}")
    return model

# =========================
# Evaluate fixed FPS (nav model)
# =========================

def evaluate_fixed_fps(fixed_fps, n_episodes=100, frame_cost=0.0):
    env = gym.make("LunarLander_VarFramerate", frame_cost=frame_cost)
    env = TimeLimit(env, max_episode_steps=500)
    
    action = fps_to_action[fixed_fps]  # always pick same FPS
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0.0

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)  # fixed action
            total_reward += reward

        episode_rewards.append(total_reward)

    env.close()
    return episode_rewards

# =========================
# Evaluate variable FPS model
# =========================
def evaluate_var_fps(var_model, n_episodes=100, frame_cost=0.0):
    env = gym.make("LunarLander_VarFramerate", frame_cost=frame_cost)
    env = TimeLimit(env, max_episode_steps=500)

    episode_rewards = []
    fps_traces_all = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0.0
        fps_trace = []

        while not (terminated or truncated):
            action, _ = var_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            fps_trace.append(info["chosen_fps"])

        episode_rewards.append(total_reward)
        fps_traces_all.append(fps_trace)

    env.close()
    return episode_rewards, fps_traces_all

# =========================
# Run multiple times and aggregate
# =========================
def run_multiple(eval_fn, n_runs, **kwargs):
    all_rewards = []
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...")
        rewards = eval_fn(**kwargs)
        if isinstance(rewards, tuple):
            rewards = rewards[0]
        all_rewards.extend(rewards)
    return np.array(all_rewards)

# =========================
# Main
# =========================
def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    var_model = load_model(VAR_MODEL_PATH)
    nav_model = load_model(NAV_MODEL_PATH)

    results = {}

    # --- Evaluate variable FPS ---
    print("\nEvaluating Variable FPS model...")
    var_rewards_all = run_multiple(
        evaluate_var_fps,
        n_runs=N_RUNS,
        var_model=var_model,
        n_episodes=N_EVAL_EPISODES,
        frame_cost=FRAME_COST
    )
    results["Variable FPS"] = var_rewards_all

    # --- Evaluate fixed FPS baselines ---
    for fixed_fps in FIXED_FPS_LIST:
        print(f"\nEvaluating Fixed FPS = {fixed_fps}...")
        fixed_rewards_all = run_multiple(
            evaluate_fixed_fps,
            n_runs=N_RUNS,
            fixed_fps=fixed_fps,
            n_episodes=N_EVAL_EPISODES,
            frame_cost=FRAME_COST
        )
        results[f"Fixed {fixed_fps} FPS"] = fixed_rewards_all

    # =========================
    # Summary
    # =========================
    print("\n===== Comparison Summary =====")
    summary = {}
    for name, rewards in results.items():
        mean = np.mean(rewards)
        std  = np.std(rewards)
        summary[name] = (mean, std)
        print(f"{name:20s} | Mean: {mean:.2f} ± {std:.2f} | N={len(rewards)}")

    # Save summary to txt
    with open(os.path.join(OUTPUT_DIR, "comparison_summary.txt"), "w") as f:
        f.write("===== Comparison Summary =====\n")
        for name, (mean, std) in summary.items():
            f.write(f"{name:20s} | Mean: {mean:.2f} ± {std:.2f} | N={N_EVAL_EPISODES * N_RUNS}\n")

    # =========================
    # Plot: Mean reward comparison bar chart
    # =========================
    names  = list(summary.keys())
    means  = [summary[n][0] for n in names]
    stds   = [summary[n][1] for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, means, yerr=stds, capsize=8, color=["steelblue", "seagreen", "darkorange"])
    plt.ylabel("Mean Episode Reward")
    plt.title(f"Reward Comparison | Frame Cost = {FRAME_COST} | {N_RUNS}x{N_EVAL_EPISODES} episodes")
    plt.grid(axis="y")
    plt.ylim(min(0, min(means) - max(stds) - 20), max(means) + max(stds) + 20)
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_bar_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # Plot: Reward distribution (boxplot)
    # =========================
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [results[n] for n in names],
        labels=names,
        patch_artist=True,
    )
    plt.ylabel("Episode Reward")
    plt.title(f"Reward Distribution | Frame Cost = {FRAME_COST} | {N_RUNS}x{N_EVAL_EPISODES} episodes")
    plt.grid(axis="y")
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_boxplot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # Plot: FPS traces (3 random episodes from var model last run)
    # =========================
    var_rewards_last, fps_traces_last = evaluate_var_fps(
        var_model, n_episodes=N_EVAL_EPISODES, frame_cost=FRAME_COST
    )
    random_indices = random.sample(range(N_EVAL_EPISODES), 3)
    for idx in random_indices:
        fps_trace = fps_traces_last[idx]
        timesteps = np.arange(1, len(fps_trace) + 1)
        plt.figure()
        plt.plot(timesteps, fps_trace, linewidth=2)
        plt.xlabel("Timestep")
        plt.ylabel("Chosen FPS")
        plt.ylim(0, 30)
        plt.title(f"Chosen FPS vs Timestep - Episode {idx+1}")
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, f"fps_trace_ep_{idx+1}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\nAll plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate()