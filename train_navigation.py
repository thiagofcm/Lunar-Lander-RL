from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import gymnasium as gym
import numpy as np
import os
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

                    #print(f"Episode {self.episode_count} Done: Env {i}, Reward: {ep_rew:.2f}, Mean Reward: {self.mean_episode_rewards[-1]:.2f}")   
        
        # TO CHECK EP LENGTH AND REWARD FROM MONITOR WRAPPER
        # for i in range(len(dones)):
        #     if dones[i]:
        #         if "episode" in infos[i]:
        #             ep_len = infos[i]["episode"]["l"]
        #             ep_rew = infos[i]["episode"]["r"]

        #             print(f"Episode finished | length={ep_len} | reward={ep_rew:.2f}")

        return True


##================================================================== TRAINING ==================================================================##

env = make_vec_env("LunarLander-v3",n_envs=N_ENV,env_kwargs={"max_episode_steps": 500})
reward_callback = EpisodeRewardCallback(n_envs=N_ENV)

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
    learning_rate= 3e-4,
)

model.learn(total_timesteps=1_500_000, callback=reward_callback)
model_name = "ppo-nav"
model.save(f"{model_dir}/{model_name}")

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