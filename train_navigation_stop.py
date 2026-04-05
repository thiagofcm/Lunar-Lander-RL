import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
import time
from datetime import datetime
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Output Settings:
current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
model_dir = f"lunar_lander_models/navigation/{current_time}"
os.makedirs(model_dir, exist_ok=True)

output_plots_dir = f"{model_dir}/plots"
os.makedirs(output_plots_dir, exist_ok=True)

# Training Settings:
N_ENV = 16

# Functions and Callback Classes:
def smooth(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="valid")

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
        #print("Step Callback Triggered")
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i in range(self.n_envs):
            if i < len(infos):

                if dones[i]:
                    ep_rew = infos[i]["episode"]["r"]

                    self.episode_rewards.append(ep_rew)
                    self.episode_count += 1
                    self.episode_idx.append(self.episode_count)
                    self.mean_episode_rewards.append(np.mean(self.episode_rewards))

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

##================================================================== TRAINING ==================================================================##

env = make_vec_env("LunarLander-v3",n_envs=N_ENV,env_kwargs={"max_episode_steps": 500})
reward_callback = EpisodeRewardCallback(n_envs=N_ENV)
convergence_callback = RewardConvergenceCallback(n_envs=N_ENV,window_size=1000,tolerance=4.0,patience=5,min_episodes=200,verbose=1,)
callback_list = CallbackList([reward_callback, convergence_callback])

# Start Training
start_time = time.time()

model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)

model.learn(total_timesteps=8_000_000, callback=callback_list)
model_name = "ppo-nav"
model.save(f"{model_dir}/{model_name}")

end_time = time.time()
training_time = (end_time - start_time)/60 # in minutes
training_total_timesteps = model.num_timesteps
training_total_episodes = reward_callback.episode_count
log_file = os.path.join(model_dir, "training_log.txt")

with open(log_file, "w") as f:
    f.write("===== TRAINING SUMMARY =====\n")
    f.write(f"Type                : Navigation\n")
    f.write(f"Model               : {model_dir}/{model_name}\n")
    f.write(f"Total timesteps     : {training_total_timesteps}\n")
    f.write(f"Total episodes      : {training_total_episodes}\n")
    f.write(f"Training time (min) : {training_time:.2f}\n")

# Plot Training Results: Reward x Episode
ep = np.array(reward_callback.episode_idx)
rew = np.array(reward_callback.episode_rewards)
rew_s = smooth(rew, window=20)
ep_s = ep[len(ep) - len(rew_s):]

plt.figure()
plt.plot(ep, rew, alpha=0.3, label="Raw")
plt.plot(ep_s, rew_s, linewidth=2, label="Smoothed")
plt.ylim(-400, 350)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Total Reward vs Episode")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_plots_dir, f"train_rew_vs_ep.png"), dpi=300, bbox_inches="tight")

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
#plt.legend()
plt.savefig(os.path.join(output_plots_dir, f"train_mean_rew_vs_ep.png"), dpi=300, bbox_inches="tight")

print(f"Training Plots saved on {output_plots_dir}")

##================================================================== EVALUATION =================================================================##
# Start Evaluation
eval_env = gym.make("LunarLander-v3", max_episode_steps=500)
n_eval_episodes = 100
episode_rewards = []

for ep in range(n_eval_episodes):
    obs, _ = eval_env.reset()
    done = False
    truncated = False
    total_reward = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward

    episode_rewards.append(total_reward)
    print(f"Evaluation Episode {ep+1}/{n_eval_episodes} | Reward: {total_reward:.2f}")

eval_env.close()

mean_reward = sum(episode_rewards) / len(episode_rewards)
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
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Evaluation Reward vs Episode")
plt.grid(True)
plt.savefig(os.path.join(output_plots_dir, f"eval_rew_vs_ep.png"), dpi=300, bbox_inches="tight")
plt.legend()

print(f"Evaluation Plots saved on {output_plots_dir}")