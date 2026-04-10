from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from sb3_contrib import RecurrentPPO
from gymnasium.wrappers import TimeLimit
from datetime import datetime
import time
import scripts.lunar_lander_var_fps as lunar_lander_var_fps
from scripts.lunar_lander_var_fps import navigation_model_path
from scripts.lunar_lander_var_fps import FPS_COST
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Functions and Callback Classes:
def smooth(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="valid")

from stable_baselines3.common.callbacks import BaseCallback


class MultiSignalConvergenceCallback(BaseCallback):
    def __init__(
        self,
        kl_threshold=0.01,
        pg_threshold=1e-3,
        ev_threshold=0.90,
        patience=5,
        check_freq=5,
        verbose=1,
    ):
        super().__init__(verbose)
        self.kl_threshold = kl_threshold
        self.pg_threshold = pg_threshold
        self.ev_threshold = ev_threshold
        self.patience = patience
        self.check_freq = check_freq

        self.stable_count = 0
        self.last_checked_update = -1

    def _on_step(self) -> bool:
        # Make sure at least one training update happened
        n_updates = self.model._n_updates

        if n_updates <= 0:
            return True

        # Only check once every check_freq updates
        if n_updates % self.check_freq != 0:
            return True

        # Avoid checking the same update multiple times
        if n_updates == self.last_checked_update:
            return True

        self.last_checked_update = n_updates

        # Latest values recorded by SB3 logger
        log_dict = self.model.logger.name_to_value

        approx_kl = log_dict.get("train/approx_kl", None)
        pg_loss = log_dict.get("train/policy_gradient_loss", None)
        explained_var = log_dict.get("train/explained_variance", None)

        # If metrics are not available yet, continue training
        if approx_kl is None or pg_loss is None or explained_var is None:
            return True

        # Check convergence condition
        converged = (
            approx_kl < self.kl_threshold
            and abs(pg_loss) < self.pg_threshold
            and explained_var > self.ev_threshold
        )

        if converged:
            self.stable_count += 1
            if self.verbose > 0:
                print(
                    f"[Convergence Check | update {n_updates}] "
                    f"KL={approx_kl:.6f}, "
                    f"|PG Loss|={abs(pg_loss):.6f}, "
                    f"EV={explained_var:.6f} "
                    f"--> stable {self.stable_count}/{self.patience}"
                )
        else:
            if self.verbose > 0:
                print(
                    f"[Convergence Check | update {n_updates}] "
                    f"KL={approx_kl:.6f}, "
                    f"|PG Loss|={abs(pg_loss):.6f}, "
                    f"EV={explained_var:.6f} "
                    f"--> reset"
                )
            self.stable_count = 0

        if self.stable_count >= self.patience:
            if self.verbose > 0:
                print("Stopping training: convergence criteria satisfied.")
            return False

        return True

class EpisodeCheckpointPlotCallback(BaseCallback):
    def __init__(
        self,
        reward_callback,
        chosen_fps_callback,
        output_root_dir,
        checkpoint_every_episodes=4000,
        smooth_window=20,
        verbose=0,
    ):
        super().__init__(verbose)
        self.reward_callback = reward_callback
        self.chosen_fps_callback = chosen_fps_callback
        self.output_root_dir = output_root_dir
        self.checkpoint_every_episodes = checkpoint_every_episodes
        self.smooth_window = smooth_window

        self.last_checkpoint_episode = 0

        os.makedirs(self.output_root_dir, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        if dones is None:
            return True

        # Count how many envs finished in this rollout step
        finished_now = int(np.sum(dones))

        if finished_now == 0:
            return True

        # Total finished episodes tracked from reward callback
        total_episodes = len(self.reward_callback.episode_idx)

        # Save checkpoint plots every N episodes
        while total_episodes >= self.last_checkpoint_episode + self.checkpoint_every_episodes:
            self.last_checkpoint_episode += self.checkpoint_every_episodes
            self._save_checkpoint_plots(self.last_checkpoint_episode)

        return True

    def _save_checkpoint_plots(self, checkpoint_ep):
        checkpoint_dir = os.path.join(
            self.output_root_dir, f"checkpoint_ep_{checkpoint_ep}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, "model")
        self.model.save(model_path)

        # =========================
        # Plot Episode Total Reward x Episode
        # =========================
        ep = np.array(self.reward_callback.episode_idx)
        rew = np.array(self.reward_callback.episode_rewards)

        print(f"=== Saving checkpoint plots at episode {checkpoint_ep} ===")
        print(f"Total episodes available: {len(ep)}")

        if len(ep) > 0 and len(rew) > 0:
            rew_s = smooth(rew, window=self.smooth_window)
            ep_s = ep[len(ep) - len(rew_s):] if len(rew_s) > 0 else np.array([])

            plt.figure()
            plt.plot(ep, rew, alpha=0.3, label="Raw")
            if len(rew_s) > 0:
                plt.plot(ep_s, rew_s, linewidth=2, label="Smoothed")
            plt.ylim(-400, 350)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title(f"Training Total Reward vs Episode (up to {checkpoint_ep})")
            plt.grid(True)
            plt.legend()
            plt.savefig(
                os.path.join(checkpoint_dir, "train_total_rew_vs_ep.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # =========================
        # Plot Mean Reward x Episode
        # =========================
        ep_mean = np.array(self.reward_callback.episode_idx)
        mean_rew = np.array(self.reward_callback.mean_episode_rewards)

        if len(ep_mean) > 0 and len(mean_rew) > 0:
            mean_rew_s = smooth(mean_rew, window=self.smooth_window)
            ep_mean_s = ep_mean[len(ep_mean) - len(mean_rew_s):] if len(mean_rew_s) > 0 else np.array([])

            plt.figure()
            plt.plot(ep_mean, mean_rew, alpha=0.3, label="Raw")
            if len(mean_rew_s) > 0:
                plt.plot(ep_mean_s, mean_rew_s, linewidth=2, label="Smoothed")
            plt.ylim(-400, 350)
            plt.xlabel("Episode")
            plt.ylabel("Mean Reward")
            plt.title(f"Mean Reward vs Episode (up to {checkpoint_ep})")
            plt.grid(True)
            plt.legend()
            plt.savefig(
                os.path.join(checkpoint_dir, "train_mean_rew_vs_ep.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # =========================
        # Plot Mean Chosen FPS x Episode
        # =========================
        ep_fps = np.array(self.chosen_fps_callback.episode_idx)
        mean_fps = np.array(self.chosen_fps_callback.episode_mean_fps)

        if len(ep_fps) > 0 and len(mean_fps) > 0:
            mean_fps_s = smooth(mean_fps, window=self.smooth_window)
            ep_fps_s = ep_fps[len(ep_fps) - len(mean_fps_s):] if len(mean_fps_s) > 0 else np.array([])

            plt.figure()
            plt.plot(ep_fps, mean_fps, alpha=0.3, label="Raw")
            if len(mean_fps_s) > 0:
                plt.plot(ep_fps_s, mean_fps_s, linewidth=2, label="Smoothed")
            plt.ylim(0, 55)
            plt.xlabel("Episode")
            plt.ylabel("Mean Chosen FPS")
            plt.title(f"Training Mean Chosen FPS vs Episode (up to {checkpoint_ep})")
            plt.grid(True)
            plt.legend()
            plt.savefig(
                os.path.join(checkpoint_dir, "train_mean_chosen_fps_vs_ep.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

class EpisodeRewardCallback(BaseCallback):
    def __init__(self, n_envs, verbose=0):
        super().__init__(verbose)
        self.n_envs = n_envs

        # accumulate reward per env
        self.current_rewards = [0.0 for _ in range(n_envs)]

        # store final episode rewards
        self.episode_rewards = []
        self.mean_episode_rewards = []
        self.episode_idx = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i in range(self.n_envs):
            if i < len(infos):

                r = infos[i].get("reward", 0.0)
                self.current_rewards[i] += r

                if dones[i]:
                    self.episode_rewards.append(self.current_rewards[i])

                    self.episode_count += 1
                    self.episode_idx.append(self.episode_count)

                    self.mean_episode_rewards.append(np.mean(self.episode_rewards))

                    self.current_rewards[i] = 0.0
        return True

class RewardConvergenceCallback(BaseCallback):
    def __init__(
        self,
        n_envs,
        window_size=100,
        tolerance=2.0,
        patience=5,
        min_episodes=200,
        verbose=1,
    ):
        super().__init__(verbose)
        self.n_envs = n_envs
        self.window_size = window_size
        self.tolerance = tolerance
        self.patience = patience
        self.min_episodes = min_episodes
        
        self.last_checked_episode = 0
        self.episode_rewards = []
        self.stable_checks = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i in range(self.n_envs):
            if i < len(infos):
                if dones[i]:
                    ep_rew = infos[i]["episode"]["r"]
                    self.episode_rewards.append(ep_rew)

        n = len(self.episode_rewards)

        # Need at least two windows to compare
        if n >= max(self.min_episodes, 2 * self.window_size):

            # ONLY check every window_size episodes
            if n % self.window_size != 0:
                return True
            if n == self.last_checked_episode:
                return True

            self.last_checked_episode = n

            recent_mean = np.mean(self.episode_rewards[-self.window_size:])
            previous_mean = np.mean(
                self.episode_rewards[-2 * self.window_size : -self.window_size]
            )

            diff = abs(recent_mean - previous_mean)

            if self.verbose > 0:
                print(
                    f"[Convergence Check] "
                    f"Episode: {n}, "
                    f"Previous mean: {previous_mean:.3f}, "
                    f"Recent mean: {recent_mean:.3f}, "
                    f"Diff: {diff:.3f}"
                )

            if diff <= self.tolerance:
                self.stable_checks += 1
                if self.verbose > 0:
                    print(
                        f"[Convergence Check] Stable count: "
                        f"{self.stable_checks}/{self.patience}"
                    )
            else:
                self.stable_checks = 0

            if self.stable_checks >= self.patience:
                if self.verbose > 0:
                    print(
                        f"Stopping training: reward mean stabilized within "
                        f"±{self.tolerance} for {self.patience} checks."
                    )
                return False

        return True

class EpisodeNavRewardCallback(BaseCallback):
    def __init__(self, n_envs, verbose=0):
        super().__init__(verbose)
        self.n_envs = n_envs

        # Running nav reward accumulator for each parallel env
        self.current_nav_rewards = [0.0 for _ in range(n_envs)]

        # Final stored values: one entry per finished episode
        self.episode_nav_rewards = []
        self.episode_idx = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals["dones"]

        for i in range(self.n_envs):
            if i < len(infos):
                self.current_nav_rewards[i] += infos[i].get("nav_reward", 0.0)

            if dones[i]:
                self.episode_nav_rewards.append(self.current_nav_rewards[i])

                self.episode_count += 1
                self.episode_idx.append(self.episode_count)

                # Reset accumulator for that env
                self.current_nav_rewards[i] = 0.0

        return True

class EpisodeFPSPenaltyCallback(BaseCallback):
    def __init__(self, n_envs, verbose=0):
        super().__init__(verbose)
        self.n_envs = n_envs

        # Running nav reward accumulator for each parallel env
        self.current_fps_penalty = [0.0 for _ in range(n_envs)]

        # Final stored values: one entry per finished episode
        self.episode_fps_penalty = []
        self.episode_idx = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals["dones"]

        for i in range(self.n_envs):
            if i < len(infos):
                self.current_fps_penalty[i] += infos[i].get("fps_penalty", 0.0)

            if dones[i]:
                self.episode_fps_penalty.append(self.current_fps_penalty[i])

                self.episode_count += 1
                self.episode_idx.append(self.episode_count)

                # Reset accumulator for that env
                self.current_fps_penalty[i] = 0.0

        return True

class EpisodeChosenFPSCallback(BaseCallback):
    def __init__(self, n_envs, verbose=0):
        super().__init__(verbose)
        self.n_envs = n_envs

        # Running accumulators per env
        self.current_chosen_fps = [0.0 for _ in range(n_envs)]
        self.current_steps = [0 for _ in range(n_envs)]

        # Final stored values
        self.episode_mean_fps = []
        self.episode_idx = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals["dones"]

        for i in range(self.n_envs):
            if i < len(infos):
                self.current_chosen_fps[i] += infos[i].get("chosen_fps", 0.0)

            self.current_steps[i] += 1

            if dones[i]:
                # Compute mean FPS
                if self.current_steps[i] > 0:
                    mean_fps = self.current_chosen_fps[i] / self.current_steps[i]
                else:
                    mean_fps = 0.0

                self.episode_mean_fps.append(mean_fps)

                self.episode_count += 1
                self.episode_idx.append(self.episode_count)

                # Reset
                self.current_chosen_fps[i] = 0.0
                self.current_steps[i] = 0

        return True

if __name__ == "__main__":

    # Output Settings:
    fps_cost_str =str(FPS_COST).replace('.', '_')

    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    model_dir = f"lunar_lander_models/var_framerate_kl/{current_time}_FPS_{fps_cost_str}"
    os.makedirs(model_dir, exist_ok=True)

    output_plots_dir = f"{model_dir}/plots"
    os.makedirs(output_plots_dir, exist_ok=True)

    # Training Settings:
    N_ENV = 16
    env = make_vec_env("LunarLander_VarFramerate", n_envs=N_ENV, env_kwargs={"max_episode_steps": 500})
    #env = gym.make("LunarLander_VarFramerate")
    model_architecture = "PPO"

    # Callback declaration
    reward_callback = EpisodeRewardCallback(n_envs=N_ENV)
    nav_reward_callback = EpisodeNavRewardCallback(n_envs=N_ENV)
    fps_penalty_callback = EpisodeFPSPenaltyCallback(n_envs=N_ENV)
    chosen_fps_callback = EpisodeChosenFPSCallback(n_envs=N_ENV)
    #convergence_callback = RewardConvergenceCallback(n_envs=N_ENV, window_size=100, tolerance=10.0, patience=10, min_episodes=200, verbose=1)
    convergence_callback = MultiSignalConvergenceCallback(kl_threshold=0.002, pg_threshold=1e-3, ev_threshold=0.90, patience=5, check_freq=5, verbose=1)
    checkpoint_plot_callback = EpisodeCheckpointPlotCallback(reward_callback=reward_callback,chosen_fps_callback=chosen_fps_callback,output_root_dir=output_plots_dir,checkpoint_every_episodes=1000,smooth_window=20,verbose=1)
    callback_list = CallbackList([reward_callback, nav_reward_callback, fps_penalty_callback, chosen_fps_callback, convergence_callback, checkpoint_plot_callback])

    # Start Training
    start_time = time.time()

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.999, 
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
        #device="cuda:2",
    )

    model.learn(total_timesteps=32_000_000, callback=callback_list)
    model_name = f"ppo_var_fps_cost_{fps_cost_str}"
    model.save(f"{model_dir}/{model_name}")

    end_time = time.time()
    training_time = (end_time - start_time)/60 # in minutes
    training_total_timesteps = model.num_timesteps
    training_total_episodes = reward_callback.episode_count
    log_file = os.path.join(model_dir, "training_log.txt")
    env.close()

    with open(log_file, "w") as f:
        f.write("===== TRAINING SUMMARY =====\n")
        f.write(f"Type                : Navigation\n")
        f.write(f"Trained Model         : {model_dir}/{model_name}\n")
        f.write(f"Navigation Model used : {navigation_model_path}\n")
        f.write(f"Total timesteps     : {training_total_timesteps}\n")
        f.write(f"Total episodes      : {training_total_episodes}\n")
        f.write(f"Training time (min) : {training_time:.2f}\n")
    
    # Plot Episode Total Reward x Episode
    ep = np.array(reward_callback.episode_idx)
    rew = np.array(reward_callback.episode_rewards)
    rew_s = smooth(rew, window=20)
    ep_s = ep[len(ep) - len(rew_s):]

    plt.figure()
    plt.plot(ep, rew, alpha=0.3, label="Raw")
    plt.plot(ep_s, rew_s, linewidth=2, label="Smoothed")
    plt.ylim(-400, 350)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Training Total Reward vs Episode - FPS COST: {FPS_COST}")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_plots_dir, f"train_total_rew_vs_ep.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Training Results: Mean Reward x Episode
    ep_mean = np.array(reward_callback.episode_idx)
    mean_rew = np.array(reward_callback.mean_episode_rewards)
    mean_rew_s = smooth(mean_rew, window=20)
    ep_mean_s = ep_mean[len(ep_mean) - len(mean_rew_s):]

    plt.figure()
    plt.plot(ep_mean, mean_rew)
    #plt.plot(ep_mean_s, mean_rew_s, linewidth=2, label="Smoothed")
    plt.ylim(-400, 350)
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward vs Episode")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_plots_dir, f"train_mean_rew_vs_ep.png"), dpi=300, bbox_inches="tight")

    # Plot Episode Nav Reward x Episode
    ep_nav = np.array(nav_reward_callback.episode_idx)
    nav_rew = np.array(nav_reward_callback.episode_nav_rewards)
    nav_rew_s = smooth(nav_rew, window=20)
    ep_nav_s = ep_nav[len(ep_nav) - len(nav_rew_s):]

    plt.figure()
    plt.plot(ep_nav, nav_rew, alpha=0.3, label="Raw")
    plt.plot(ep_nav_s, nav_rew_s, linewidth=2, label="Smoothed")
    plt.xlabel("Episode")
    plt.ylabel("Nav Reward")
    plt.ylim(-400, 350)
    plt.title(f"Training Nav Reward vs Episode - FPS COST: {FPS_COST}")
    plt.grid(True)
    plt.savefig(os.path.join(output_plots_dir, f"train_nav_rew_vs_ep.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Fps Penalty x Episode
    ep_fps_pen = np.array(fps_penalty_callback.episode_idx)
    fps_pen = np.array(fps_penalty_callback.episode_fps_penalty)
    fps_pen_s = smooth(fps_pen, window=20)
    ep_fps_pen_s = ep_fps_pen[len(ep_fps_pen) - len(fps_pen_s):]

    plt.figure()
    plt.plot(ep_fps_pen, fps_pen, alpha=0.3, label="Raw")
    plt.plot(ep_fps_pen_s, fps_pen_s, linewidth=2, label="Smoothed")
    plt.ylim(-400, 350)
    plt.xlabel("Episode")
    plt.ylabel("FPS Penalty")
    plt.title(f"Training FPS Penalty vs Episode - FPS COST: {FPS_COST}")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_plots_dir, f"train_fps_penalty_vs_ep.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Mean Chosen Fps x Episode
    ep_fps = np.array(chosen_fps_callback.episode_idx)
    mean_fps = np.array(chosen_fps_callback.episode_mean_fps)
    mean_fps_s = smooth(mean_fps, window=20)
    ep_fps_s = ep_fps[len(ep_fps) - len(mean_fps_s):]

    plt.figure()
    plt.plot(ep_fps, mean_fps, alpha=0.3, label="Raw")
    plt.plot(ep_fps_s, mean_fps_s, linewidth=2, label="Smoothed")
    plt.ylim(0, 55)
    plt.xlabel("Episode")
    plt.ylabel("Mean Chosen FPS")
    plt.title(f"Training Mean Chosen FPS vs Episode - FPS COST: {FPS_COST}")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_plots_dir, f"train_mean_chosen_fps_vs_ep.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training Plots saved on {output_plots_dir}")

    ##==================== EVALUATION ====================##

    import random

    # Start Evaluation
    eval_env = gym.make("LunarLander_VarFramerate")
    eval_env = TimeLimit(eval_env, max_episode_steps=500)

    #model = RecurrentPPO.load("lunar_lander_models/var_framerate_kl/07-04-2026_17-13-02_FPS_0_1/ppo_var_fps_cost_0_1.zip")
    #output_plots_dir = "lunar_lander_models/var_framerate_kl/07-04-2026_17-13-02_FPS_0_1/plots"
    n_eval_episodes = 100
    episode_rewards = []
    mean_chosen_fps_eval = []

    # Store FPS-vs-timestep traces for all episodes first
    fps_traces_all = []

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
        print(f"Evaluation Episode {ep+1}/{n_eval_episodes} | Reward: {total_reward:.2f}")
        mean_chosen_fps_eval.append(np.mean(fps_trace))
        fps_traces_all.append(fps_trace)

    eval_env.close()

    mean_reward = np.mean(episode_rewards)
    print(f"mean_reward={mean_reward:.2f}")

    # Plot Evaluation Results: Reward x Episode
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
    plt.savefig(os.path.join(output_plots_dir, f"eval_total_rew_vs_ep.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Evaluation Mean Chosen FPS x Episode
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
    plt.savefig(os.path.join(output_plots_dir, f"eval_mean_chosen_fps_vs_ep.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Select 3 random episodes
    random_episode_indices = random.sample(range(n_eval_episodes), 3)

    for idx in random_episode_indices:
        fps_trace = fps_traces_all[idx]
        timesteps = np.arange(1, len(fps_trace) + 1)

        plt.figure()
        plt.plot(timesteps, fps_trace, linewidth=2)
        plt.xlabel("Timestep")
        plt.ylabel("Chosen FPS")
        plt.ylim(0, 55)
        plt.title(f"Chosen FPS vs Timestep - Evaluation Episode {idx + 1}")
        plt.savefig(os.path.join(output_plots_dir, f"chosen_fps_vs_timestep_ep_{idx+1}.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    print(f"Evaluation Plots saved on {output_plots_dir}")