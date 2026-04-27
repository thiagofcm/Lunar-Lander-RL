import os
import re
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
import scripts.lunar_lander_var_fps as lunar_lander_var_fps
from scripts.lunar_lander_var_fps import LANDING_PENALTY

# =========================
# USER INPUT
# =========================
ROOT_DIR   = "lunar_lander_models\\var_framerate_per_frame_penalty_wo_landing_penalty"
N_EPISODES = 100
RUN_SEED   = 42

# =========================
# Evaluate a single model
# =========================
def evaluate_model(model, frame_cost, n_episodes, seeds):
    env = gym.make("LunarLander_VarFramerate", frame_cost=frame_cost)
    env = TimeLimit(env, max_episode_steps=500)

    rewards = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)                                                                                                           
        terminated, truncated = False, False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return np.mean(rewards), np.std(rewards)

# =========================
# Find best checkpoint for one run
# =========================
def find_best_checkpoint(run_dir, frame_cost, seeds):
    plots_dir = os.path.join(run_dir, "plots")

    # Find all checkpoint dirs sorted numerically
    checkpoint_dirs = sorted([
        d for d in os.listdir(plots_dir)
        if d.startswith("checkpoint_ep_")
    ], key=lambda x: int(x.split("_")[-1]))

    print(f"\n  Found {len(checkpoint_dirs)} checkpoints")

    best_mean       = -np.inf
    best_std        = None
    best_checkpoint = None
    best_path       = None
    all_results     = []

    for ckpt_dir in checkpoint_dirs:
        # Find the zip file inside the checkpoint dir
        ckpt_path = os.path.join(plots_dir, ckpt_dir)
        zip_files = [f for f in os.listdir(ckpt_path) if f.endswith(".zip")]

        if not zip_files:
            print(f"    Skipping {ckpt_dir} — no .zip found")
            continue

        model_path = os.path.join(ckpt_path, zip_files[0])

        try:
            model = PPO.load(model_path)
        except Exception as e:
            print(f"    Skipping {ckpt_dir} — failed to load: {e}")
            continue

        mean_r, std_r = evaluate_model(model, frame_cost, len(seeds), seeds)
        all_results.append((ckpt_dir, mean_r, std_r, model_path))
        print(f"    {ckpt_dir:25s} | Mean: {mean_r:7.2f} ± {std_r:6.2f}")

        if mean_r > best_mean:
            best_mean       = mean_r
            best_std        = std_r
            best_checkpoint = ckpt_dir
            best_path       = model_path

    return best_checkpoint, best_mean, best_std, best_path, all_results

# =========================
# Main — iterate all runs
# =========================
if __name__ == "__main__":

    seeds = [RUN_SEED + i for i in range(N_EPISODES)]

    # Discover all run directories and extract frame cost from folder name
    run_dirs = {}
    for folder in sorted(os.listdir(ROOT_DIR)):
        match = re.search(r"FPS_(.+)$", folder)
        if match:
            frame_cost_str = match.group(1).replace("_", ".")
            frame_cost = float(frame_cost_str)
            run_dirs[frame_cost] = os.path.join(ROOT_DIR, folder)

    print(f"Found {len(run_dirs)} runs:")
    for fc, path in sorted(run_dirs.items()):
        print(f"  Frame Cost {fc} → {path}")

    # Results summary
    summary = {}

    for frame_cost, run_dir in sorted(run_dirs.items()):
        print(f"\n{'='*60}")
        print(f"Frame Cost: {frame_cost} | Run: {run_dir}")

        best_ckpt, best_mean, best_std, best_path, all_results = find_best_checkpoint(
            run_dir=run_dir,
            frame_cost=frame_cost,
            seeds=seeds
        )

        summary[frame_cost] = {
            "best_checkpoint": best_ckpt,
            "best_mean":       best_mean,
            "best_std":        best_std,
            "best_path":       best_path,
        }

        print(f"\nBest: {best_ckpt} | Mean: {best_mean:.2f} ± {best_std:.2f}")
        print(f"Path: {best_path}")

    # =========================
    # Final Summary
    # =========================
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — Best checkpoint per frame cost")
    print(f"{'='*60}")
    print(f"{'Frame Cost':>12} | {'Best Checkpoint':>25} | {'Mean Reward':>12} | {'Std':>8}")
    print("-" * 65)

    for fc in sorted(summary.keys()):
        s = summary[fc]
        print(
            f"{fc:>12.2f} | "
            f"{s['best_checkpoint']:>25} | "
            f"{s['best_mean']:>12.2f} | "
            f"{s['best_std']:>8.2f}"
        )

    # Save summary
    summary_path = os.path.join(ROOT_DIR, f"best_checkpoints_touchdown_{LANDING_PENALTY}.txt")
    with open(summary_path, "w") as f:
        f.write("Best checkpoint per frame cost\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'Frame Cost':>12} | {'Best Checkpoint':>25} | {'Mean Reward':>12} | {'Std':>8}\n")
        f.write("-" * 65 + "\n")
        for fc in sorted(summary.keys()):
            s = summary[fc]
            f.write(
                f"{fc:>12.2f} | "
                f"{s['best_checkpoint']:>25} | "
                f"{s['best_mean']:>12.2f} | "
                f"{s['best_std']:>8.2f}\n"
            )
        f.write("\nFull paths:\n")
        for fc in sorted(summary.keys()):
            f.write(f"  {fc}: {summary[fc]['best_path']}\n")

    print(f"\nSummary saved to: {summary_path}")