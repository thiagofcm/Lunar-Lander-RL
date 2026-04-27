import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
import scripts.lunar_lander_var_fps as lunar_lander_var_fps
import re


# =========================
# USER INPUT
# =========================
MODEL_PATHS_TXT = "models_to_evaluate_wo_landing_penalty.txt"  # one model path per line
OUTPUT_DIR      = "ablation_evaluation_results"
N_EPISODES      = 100
N_RUNS          = 10
RUN_SEED        = 42
LANDING_PENALTY = lunar_lander_var_fps.LANDING_PENALTY

# =========================
# Utils
# =========================
def get_seeds(run, n_episodes):
    return [RUN_SEED + run * n_episodes + i for i in range(n_episodes)]

def load_model(path):
    model = PPO.load(path)
    print(f"  Loaded: {path}")
    return model

def extract_frame_cost(model_path):
    match = re.search(r"FPS_(\d+_\d+)", model_path)
    if match:
        return float(match.group(1).replace("_", "."))
    raise ValueError(f"Could not extract frame cost from path: {model_path}")

# =========================
# Evaluate one model for one run
# =========================
def evaluate_model_single_run(model, seeds, frame_cost):
    env = gym.make("LunarLander_VarFramerate", frame_cost=frame_cost)
    env = TimeLimit(env, max_episode_steps=500)

    episode_rewards      = []
    episode_nav_rewards  = []
    episode_frames       = []
    episode_vy           = []
    episode_success      = []
    episode_fps_traces   = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        terminated, truncated = False, False

        total_reward     = 0.0
        total_nav_reward = 0.0
        frame_count      = 0
        fps_trace        = []
        touchdown_vy     = None
        landed           = False
        touchdown_flag   = False
        went_up_after    = False
        landed_in_flags  = False
        outside_flags_after_landing = False
        prev_leg_contact = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            true_obs = env.unwrapped.current_obs
            #print(("True Obs: ", true_obs))
            
            total_reward     += reward
            total_nav_reward += info.get("nav_reward", 0.0)
            frame_count       = info["episode_frame_count"]
            fps_trace.append(info["chosen_fps"])

            leg_contact = bool(true_obs[6] or true_obs[7])

            # Detect first touchdown
            if leg_contact and not touchdown_flag:
                touchdown_vy   = abs(true_obs[3])
                touchdown_flag = True
                landed         = True
                landed_in_flags = (-0.2 < true_obs[0] < 0.2)

            # While grounded, check if it drifts outside flags
            if touchdown_flag and leg_contact:
                if not (-0.2 < true_obs[0] < 0.2):
                    outside_flags_after_landing = True

            # Detect bounce
            if touchdown_flag and prev_leg_contact and not leg_contact:
                went_up_after = True

            prev_leg_contact = leg_contact

        # Successful = landed in flags AND never drifted outside AND never bounced
        successful = landed_in_flags and not outside_flags_after_landing and not went_up_after

        episode_rewards.append(total_reward)
        episode_nav_rewards.append(total_nav_reward)
        episode_frames.append(frame_count)
        episode_vy.append(touchdown_vy if touchdown_vy is not None else np.nan)
        episode_success.append(float(successful))
        episode_fps_traces.append(fps_trace)

    env.close()

    return {
        "rewards":      np.array(episode_rewards),
        "nav_rewards":  np.array(episode_nav_rewards),
        "frames":       np.array(episode_frames),
        "vy":           np.array(episode_vy),
        "success":      np.array(episode_success),
        "fps_traces":   episode_fps_traces,
    }

# =========================
# Evaluate one model across all runs
# =========================
def evaluate_model_full(model, frame_cost, n_runs, n_episodes):
    all_rewards     = []
    all_nav_rewards = []
    all_frames      = []
    all_vy          = []
    all_success     = []
    all_fps_traces  = []

    for run in range(n_runs):
        seeds = get_seeds(run, n_episodes)
        print(f"    Run {run+1}/{n_runs} | Seeds {seeds[0]}..{seeds[-1]}")

        result = evaluate_model_single_run(model, seeds, frame_cost)

        all_rewards.extend(result["rewards"])
        all_nav_rewards.extend(result["nav_rewards"])
        all_frames.extend(result["frames"])
        all_vy.extend(result["vy"])
        all_success.extend(result["success"])
        all_fps_traces.extend(result["fps_traces"])

    return {
        "rewards":     np.array(all_rewards),
        "nav_rewards": np.array(all_nav_rewards),
        "frames":      np.array(all_frames),
        "vy":          np.array(all_vy),
        "success":     np.array(all_success),
        "fps_traces":  all_fps_traces,
    }

# =========================
# Main
# =========================
if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read model paths from txt file
    with open(MODEL_PATHS_TXT, "r") as f:
        model_paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Found {len(model_paths)} models to evaluate")

    all_results = {}
    summary_lines = []
    summary_lines.append("===== Ablation Evaluation Summary  =====")
    summary_lines.append(f"Frame Cost: 0.0 - 3.0 | Landing Penalty: {LANDING_PENALTY} | Runs: {N_RUNS} | Episodes per run: {N_EPISODES}")
    summary_lines.append(f"Total episodes per model: {N_RUNS * N_EPISODES}")
    summary_lines.append("=" * 90)
    summary_lines.append(
        f"{'Model':<50} | {'Nav Rew':>10} | {'Frames':>8} | {'vy':>8} | {'Success%':>9}"
    )
    summary_lines.append("-" * 90)

    for model_path in model_paths:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_path}")
        frame_cost = extract_frame_cost(model_path)
        print(f"  Frame cost: {frame_cost}")

        try:
            model = load_model(model_path)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        result = evaluate_model_full(
            model=model,
            frame_cost=frame_cost,
            n_runs=N_RUNS,
            n_episodes=N_EPISODES,
        )

        # Compute summary stats
        mean_reward     = np.mean(result["rewards"])
        std_reward      = np.std(result["rewards"])
        mean_nav_reward = np.mean(result["nav_rewards"])
        std_nav_reward  = np.std(result["nav_rewards"])
        mean_frames     = np.mean(result["frames"])
        std_frames      = np.std(result["frames"])
        mean_vy         = np.nanmean(result["vy"])
        std_vy          = np.nanstd(result["vy"])
        success_rate    = np.mean(result["success"]) * 100

        print(f"\n  Results:")
        print(f"  Total Reward:  {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Nav Reward:    {mean_nav_reward:.2f} ± {std_nav_reward:.2f}")
        print(f"  Frames:        {mean_frames:.1f} ± {std_frames:.1f}")
        print(f"  vy touchdown:  {mean_vy:.4f} ± {std_vy:.4f}")
        print(f"  Success rate:  {success_rate:.1f}%")

        # Store results
        model_name = os.path.basename(os.path.dirname(model_path))
        all_results[model_path] = {
            "model_name":    model_name,
            "mean_reward":   mean_reward,
            "std_reward":    std_reward,
            "mean_nav":      mean_nav_reward,
            "std_nav":       std_nav_reward,
            "mean_frames":   mean_frames,
            "std_frames":    std_frames,
            "mean_vy":       mean_vy,
            "std_vy":        std_vy,
            "success_rate":  success_rate,
            "raw":           result,
        }

        summary_lines.append(
            f"{model_name:<50} | "
            f"{mean_nav_reward:>7.2f}±{std_nav_reward:<6.2f} | "
            f"{mean_frames:>5.1f}±{std_frames:<5.1f} | "
            f"{mean_vy:>6.4f}±{std_vy:<6.4f} | "
            f"{success_rate:>8.1f}% | "
        )

        # Save per-model npy
        npy_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.npy")
        np.save(npy_path, result)
        print(f"  Saved: {npy_path}")

    # =========================
    # Save summary txt
    # =========================
    summary_path = os.path.join(OUTPUT_DIR, "ablation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSummary saved to: {summary_path}")

    # =========================
    # Save all results as npy
    # =========================
    all_results_path = os.path.join(OUTPUT_DIR, "all_results.npy")
    np.save(all_results_path, all_results)
    print(f"All results saved to: {all_results_path}")

    # Print final summary
    print("\n")
    for line in summary_lines:
        print(line)