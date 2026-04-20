# =========================
# Imports
# =========================
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit

import scripts.lunar_lander_var_fps as lunar_lander_var_fps
from scripts.lunar_lander_var_fps import navigation_model_path

# =========================
# USER INPUT
# =========================
VAR_MODEL_PATH  = "lunar_lander_models\\var_framerate_per_frame_penalty\\16-04-2026_09-01-55_FPS_0_6\\plots\\checkpoint_ep_16000\\model.zip"
OUTPUT_DIR      = "comparison_output"
FRAME_COST      = 0.6
N_EVAL_EPISODES = 100
N_RUNS          = 10
RUN_SEED        = 42
FIXED_FPS_LIST  = [1, 5, 10, 25]
fps_to_action   = {1: 0, 5: 1, 10: 2, 25: 3}

# =========================
# Utils
# =========================
def smooth(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def load_model(path):
    model = PPO.load(path)
    print(f"Loaded PPO: {path}")
    return model

# =========================
# Evaluate ONE condition for ONE run
# given a fixed list of seeds
# =========================
def evaluate_fixed_fps(fixed_fps, seeds, frame_cost=0.0):
    env = gym.make("LunarLander_VarFramerate", frame_cost=frame_cost)
    env = TimeLimit(env, max_episode_steps=500)

    action = fps_to_action[fixed_fps]
    episode_rewards = []
    episode_frames  = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        terminated, truncated = False, False
        total_reward = 0.0
        frame_count  = 0

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frame_count   = info["episode_frame_count"]

        episode_rewards.append(total_reward)
        episode_frames.append(frame_count)

    env.close()
    return episode_rewards, episode_frames


def evaluate_var_fps(var_model, seeds, frame_cost=0.0):
    env = gym.make("LunarLander_VarFramerate", frame_cost=frame_cost)
    env = TimeLimit(env, max_episode_steps=500)

    episode_rewards   = []
    episode_frames    = []
    fps_traces_all    = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        terminated, truncated = False, False
        total_reward = 0.0
        frame_count  = 0
        fps_trace    = []

        while not (terminated or truncated):
            action, _ = var_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frame_count   = info["episode_frame_count"]
            fps_trace.append(info["chosen_fps"])

        episode_rewards.append(total_reward)
        episode_frames.append(frame_count)
        fps_traces_all.append(fps_trace)

    env.close()
    return episode_rewards, episode_frames, fps_traces_all


# =========================
# Main
# =========================
def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    var_model = load_model(VAR_MODEL_PATH)

    # results[condition] = {"rewards": [], "frames": []}
    conditions = ["Variable FPS"] + [f"Fixed {fps} FPS" for fps in FIXED_FPS_LIST]
    results = {c: {"rewards": [], "frames": []} for c in conditions}
    all_fps_traces = []

    # =========================
    # Run N_RUNS times
    # each run uses the SAME seeds for ALL conditions
    # =========================
    for run in range(N_RUNS):
        # Generate seeds for this run — shared across all conditions
        seeds = [RUN_SEED + run * N_EVAL_EPISODES + i for i in range(N_EVAL_EPISODES)]
        print(f"\n===== Run {run+1}/{N_RUNS} | Seeds {seeds[0]}..{seeds[-1]} =====")

        # Variable FPS
        print("  Evaluating Variable FPS...")
        var_r, var_f, var_traces = evaluate_var_fps(var_model, seeds=seeds, frame_cost=FRAME_COST)
        results["Variable FPS"]["rewards"].extend(var_r)
        results["Variable FPS"]["frames"].extend(var_f)
        if run == N_RUNS - 1:  # save traces from last run for plotting
            all_fps_traces = var_traces

        # Fixed FPS baselines — same seeds
        for fixed_fps in FIXED_FPS_LIST:
            print(f"  Evaluating Fixed {fixed_fps} FPS...")
            fix_r, fix_f = evaluate_fixed_fps(fixed_fps, seeds=seeds, frame_cost=FRAME_COST)
            results[f"Fixed {fixed_fps} FPS"]["rewards"].extend(fix_r)
            results[f"Fixed {fixed_fps} FPS"]["frames"].extend(fix_f)

    # =========================
    # Summary
    # =========================
    print("\n===== Comparison Summary =====")
    summary = {}
    for name in conditions:
        rewards = np.array(results[name]["rewards"])
        frames  = np.array(results[name]["frames"])
        summary[name] = {
            "mean_reward": np.mean(rewards),
            "std_reward":  np.std(rewards),
            "mean_frames": np.mean(frames),
            "std_frames":  np.std(frames),
        }
        print(
            f"{name:20s} | "
            f"Reward: {summary[name]['mean_reward']:7.2f} ± {summary[name]['std_reward']:6.2f} | "
            f"Frames: {summary[name]['mean_frames']:6.1f} ± {summary[name]['std_frames']:5.1f} | "
            f"N={len(rewards)}"
        )

    with open(os.path.join(OUTPUT_DIR, "comparison_summary.txt"), "w") as f:
        f.write("===== Comparison Summary =====\n")
        f.write(f"Frame Cost: {FRAME_COST} | Runs: {N_RUNS} | Episodes per run: {N_EVAL_EPISODES}\n\n")
        for name in conditions:
            s = summary[name]
            f.write(
                f"{name:20s} | "
                f"Reward: {s['mean_reward']:7.2f} ± {s['std_reward']:6.2f} | "
                f"Frames: {s['mean_frames']:6.1f} ± {s['std_frames']:5.1f}\n"
            )

    # =========================
    # Plots
    # =========================
    names   = conditions
    means_r = [summary[n]["mean_reward"] for n in names]
    stds_r  = [summary[n]["std_reward"]  for n in names]
    means_f = [summary[n]["mean_frames"] for n in names]
    stds_f  = [summary[n]["std_frames"]  for n in names]
    colors  = ["steelblue", "seagreen", "darkorange", "firebrick", "purple"]

    # Reward bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(names, means_r, yerr=stds_r, capsize=8, color=colors)
    plt.ylabel("Mean Episode Reward")
    plt.title(f"Reward Comparison | Frame Cost={FRAME_COST} | {N_RUNS}x{N_EVAL_EPISODES} eps (same seeds)")
    plt.grid(axis="y")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_reward_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Frames bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(names, means_f, yerr=stds_f, capsize=8, color=colors)
    plt.ylabel("Mean Frames Acquired per Episode")
    plt.title(f"Frames Acquired | Frame Cost={FRAME_COST} | {N_RUNS}x{N_EVAL_EPISODES} eps (same seeds)")
    plt.grid(axis="y")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_frames_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Reward boxplot
    plt.figure(figsize=(10, 5))
    plt.boxplot([results[n]["rewards"] for n in names], labels=names, patch_artist=True)
    plt.ylabel("Episode Reward")
    plt.title(f"Reward Distribution | Frame Cost={FRAME_COST} | {N_RUNS}x{N_EVAL_EPISODES} eps (same seeds)")
    plt.grid(axis="y")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_reward_boxplot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Frames boxplot
    plt.figure(figsize=(10, 5))
    plt.boxplot([results[n]["frames"] for n in names], labels=names, patch_artist=True)
    plt.ylabel("Frames Acquired per Episode")
    plt.title(f"Frames Distribution | Frame Cost={FRAME_COST} | {N_RUNS}x{N_EVAL_EPISODES} eps (same seeds)")
    plt.grid(axis="y")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_frames_boxplot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Reward vs Frames scatter
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(names):
        rewards = results[name]["rewards"]
        frames  = results[name]["frames"]
        plt.scatter(frames, rewards, alpha=0.2, label=name, color=colors[i], s=10)
        plt.scatter(np.mean(frames), np.mean(rewards), color=colors[i], s=150, marker="X", zorder=5)
    plt.xlabel("Frames Acquired")
    plt.ylabel("Episode Reward")
    plt.title("Reward vs Frames Acquired (X = mean)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reward_vs_frames_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # FPS traces (3 random episodes from last run)
    if all_fps_traces:
        random_indices = random.sample(range(len(all_fps_traces)), 3)
        for idx in random_indices:
            fps_trace = all_fps_traces[idx]
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