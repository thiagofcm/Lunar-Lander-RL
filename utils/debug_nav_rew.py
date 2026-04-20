import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
import sys
from gymnasium.wrappers import TimeLimit
import scripts.lunar_lander_nav as lunar_lander_nav

# ============================================================
# Helpers
# ============================================================

def smooth(data, window=20):
    data = np.asarray(data, dtype=np.float32)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def evaluate_fixed_sampling_fps(
    model,
    model_str,
    fps_choices=(1, 5, 10, 25, 50),
    n_eval_episodes=100,
    simulation_fps=50,
    max_episode_steps=500,
    output_root="eval_fixed_sampling",
    render_mode=None,
):
    timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
    output_dir = os.path.join(output_root, f"fixed_sampling_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    summary_results = {}

    for fixed_fps in fps_choices:
        if simulation_fps % fixed_fps != 0:
            raise ValueError(
                f"simulation_fps={simulation_fps} must be divisible by fixed_fps={fixed_fps}"
            )

        obs_interval = int(simulation_fps / fixed_fps)
        fps_dir = os.path.join(output_dir, f"fps_{fixed_fps}")
        os.makedirs(fps_dir, exist_ok=True)
        log_file = os.path.join(fps_dir, "evaluation_log.txt")

        print("\n" + "=" * 70)
        print(f"Evaluating fixed sampling FPS = {fixed_fps} | obs_interval = {obs_interval}")
        print("=" * 70)

        eval_env = gym.make(
            "LunarLander_Nav",
            #max_episode_steps=max_episode_steps,
            render_mode=render_mode
        )

        eval_env = TimeLimit(eval_env, max_episode_steps=max_episode_steps)

        episode_rewards = []
        episode_lengths = []
        all_step_rewards = []
        sampled_frames_per_episode = []

        for ep in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            step_count = 0
            step_rewards = []

            sampled_obs = np.array(obs, dtype=np.float32).copy()
            sampled_frames = 1  # count reset sample as first available frame

            while not (done or truncated):
                if step_count % obs_interval == 0:
                    sampled_obs = np.array(obs, dtype=np.float32).copy()
                    if step_count > 0:
                        sampled_frames += 1

                action, _ = model.predict(sampled_obs, deterministic=True)

                obs, reward, done, truncated, info = eval_env.step(action)
                step_rewards.append(reward)
                total_reward += reward
                step_count += 1

            # Record episode metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            sampled_frames_per_episode.append(sampled_frames)
            all_step_rewards.append(step_rewards)

            print(
                f"FPS {fixed_fps:>2} | Episode {ep+1:>3}/{n_eval_episodes} | "
                f"Reward: {total_reward:>8.2f} | Steps: {step_count:>3} | "
                f"Sampled Frames: {sampled_frames:>3}"
            )

        eval_env.close()

        # Metrics
        rewards_arr = np.array(episode_rewards, dtype=np.float32)
        lengths_arr = np.array(episode_lengths, dtype=np.float32)
        sampled_arr = np.array(sampled_frames_per_episode, dtype=np.float32)

        mean_reward = float(np.mean(rewards_arr))
        std_reward = float(np.std(rewards_arr))
        mean_length = float(np.mean(lengths_arr))
        mean_sampled = float(np.mean(sampled_arr))

        # Mean step reward across episodes
        max_len = max(len(r) for r in all_step_rewards)
        reward_matrix = np.full((len(all_step_rewards), max_len), np.nan, dtype=np.float32)

        for i, r in enumerate(all_step_rewards):
            reward_matrix[i, :len(r)] = r

        mean_step_reward = np.nanmean(reward_matrix, axis=0)

        summary_results[fixed_fps] = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "sampled_frames_per_episode": sampled_frames_per_episode,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "mean_sampled_frames": mean_sampled,
            "obs_interval": obs_interval,
            "mean_step_reward": mean_step_reward,
        }

        with open(log_file, "w") as f:
            f.write("===== TRAINING SUMMARY =====\n")
            f.write(f"Type                : Navigation\n")
            f.write(f"Model               : {model_str}\n")
            f.write(f"Mean Reward         : {summary_results[fixed_fps]['mean_reward']}\n")
            f.write(f"Std Reward          : {summary_results[fixed_fps]['std_reward']}\n")
            f.write(f"Mean Length         : {summary_results[fixed_fps]['mean_length']}\n")
            f.write(f"Total episodes      : {len(summary_results[fixed_fps]['episode_rewards'])}\n")
            f.write(f"Mean Sampled Frames : {summary_results[fixed_fps]['mean_sampled_frames']}\n")
            f.write(f"Observation Interval: {summary_results[fixed_fps]['obs_interval']} steps\n")

        print(
            f"\nFinished FPS={fixed_fps} | mean_reward={mean_reward:.2f} | "
            f"std_reward={std_reward:.2f} | mean_steps={mean_length:.2f} | "
            f"mean_sampled_frames={mean_sampled:.2f}"
        )

        # ------------------------------------------------------------
        # Plot 1: Reward vs Episode
        # ------------------------------------------------------------
        ep_eval = np.arange(1, len(episode_rewards) + 1)
        rew_eval = rewards_arr
        rew_eval_s = smooth(rew_eval, window=20)
        ep_eval_s = ep_eval[len(ep_eval) - len(rew_eval_s):]

        plt.figure()
        plt.plot(ep_eval, rew_eval, alpha=0.3, label="Raw")
        plt.plot(ep_eval_s, rew_eval_s, linewidth=2, label="Smoothed")
        plt.ylim(-400, 350)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Evaluation Reward vs Episode | Fixed Sampling FPS = {fixed_fps}")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(fps_dir, f"eval_rew_vs_ep_fps_{fixed_fps}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # ------------------------------------------------------------
        # Plot 2: Sampled frames vs Episode
        # ------------------------------------------------------------
        sampled_s = smooth(sampled_arr, window=20)
        ep_sampled_s = ep_eval[len(ep_eval) - len(sampled_s):]

        plt.figure()
        plt.plot(ep_eval, sampled_arr, alpha=0.3, label="Raw")
        plt.plot(ep_sampled_s, sampled_s, linewidth=2, label="Smoothed")
        plt.xlabel("Episode")
        plt.ylabel("Sampled Frames per Episode")
        plt.title(f"Sampled Frames vs Episode | Fixed Sampling FPS = {fixed_fps}")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(fps_dir, f"sampled_frames_vs_ep_fps_{fixed_fps}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # ------------------------------------------------------------
        # Plot 3: Mean step reward across episodes
        # ------------------------------------------------------------
        plt.figure()
        plt.plot(mean_step_reward)
        plt.xlabel("Timestep")
        plt.ylabel("Mean Reward")
        plt.title(f"Mean Step Reward | Fixed Sampling FPS = {fixed_fps}")
        plt.grid(True)
        plt.savefig(
            os.path.join(fps_dir, f"mean_step_reward_fps_{fixed_fps}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # ------------------------------------------------------------
        # Save raw arrays
        # ------------------------------------------------------------
        # np.save(os.path.join(fps_dir, f"episode_rewards_fps_{fixed_fps}.npy"), rewards_arr)
        # np.save(os.path.join(fps_dir, f"episode_lengths_fps_{fixed_fps}.npy"), lengths_arr)
        # np.save(os.path.join(fps_dir, f"sampled_frames_fps_{fixed_fps}.npy"), sampled_arr)
        # np.save(os.path.join(fps_dir, f"mean_step_reward_fps_{fixed_fps}.npy"), mean_step_reward)
        # np.save(os.path.join(fps_dir, f"step_rewards_matrix_fps_{fixed_fps}.npy"), reward_matrix)

    # # ============================================================
    # # Combined comparison plots
    # # ============================================================
    tested_fps = list(summary_results.keys())
    # mean_rewards = [summary_results[f]["mean_reward"] for f in tested_fps]
    # std_rewards = [summary_results[f]["std_reward"] for f in tested_fps]
    # mean_sampled_frames = [summary_results[f]["mean_sampled_frames"] for f in tested_fps]

    # # Combined plot: mean reward vs fixed FPS
    # plt.figure()
    # plt.plot(tested_fps, mean_rewards, marker="o")
    # plt.xlabel("Fixed Sampling FPS")
    # plt.ylabel("Mean Episode Reward")
    # plt.title("Mean Reward vs Fixed Sampling FPS")
    # plt.grid(True)
    # plt.savefig(
    #     os.path.join(output_dir, "mean_reward_vs_fixed_fps.png"),
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.close()

    # # Combined plot: mean sampled frames vs fixed FPS
    # plt.figure()
    # plt.plot(tested_fps, mean_sampled_frames, marker="o")
    # plt.xlabel("Fixed Sampling FPS")
    # plt.ylabel("Mean Sampled Frames per Episode")
    # plt.title("Mean Sampled Frames vs Fixed Sampling FPS")
    # plt.grid(True)
    # plt.savefig(
    #     os.path.join(output_dir, "mean_sampled_frames_vs_fixed_fps.png"),
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.close()

    # # Combined plot: mean step reward comparison
    # plt.figure()
    # for fps in tested_fps:
    #     plt.plot(summary_results[fps]["mean_step_reward"], label=f"FPS {fps}")
    # plt.xlabel("Timestep")
    # plt.ylabel("Mean Reward")
    # plt.title("Mean Step Reward Comparison Across Fixed FPS")
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(
    #     os.path.join(output_dir, "mean_step_reward_comparison.png"),
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.close()

    # Save summary text
    summary_txt = os.path.join(output_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Fixed Sampling FPS Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"n_eval_episodes: {n_eval_episodes}\n")
        f.write(f"simulation_fps: {simulation_fps}\n")
        f.write(f"max_episode_steps: {max_episode_steps}\n\n")

        for fps in tested_fps:
            r = summary_results[fps]
            f.write(
                f"FPS={fps:>2} | obs_interval={r['obs_interval']:>2} | "
                f"mean_reward={r['mean_reward']:.2f} | std_reward={r['std_reward']:.2f} | "
                f"mean_steps={r['mean_length']:.2f} | "
                f"mean_sampled_frames={r['mean_sampled_frames']:.2f}\n"
            )

    print(f"\nAll evaluation plots and results saved in:\n{output_dir}")
    return summary_results

if __name__ == "__main__":

    model_str = sys.argv[1]
    model = PPO.load(model_str)
    print(f"Model Loaded: {model_str}")

    summary = evaluate_fixed_sampling_fps(
        model=model,
        model_str=model_str,
        fps_choices=(1, 5, 10, 25, 50),
        n_eval_episodes=100,
        simulation_fps=50,
        max_episode_steps=500,
        #output_root=output_plots_dir,
        render_mode=None,
    )