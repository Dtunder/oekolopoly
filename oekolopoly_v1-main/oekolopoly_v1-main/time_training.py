"""Script to time wall-time of training Oekolopoly
Combination: No observation wrapper, Box action wrapper, and PerRound
reward wrapper with R_c=0.5 as these have comparable round numbers
among algorithms
"""

import time
from pathlib import Path
import gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
import oekolopoly.oekolopoly
from oekolopoly.wrappers import OekoBoxActionWrapper, OekoPerRoundRewardWrapper


ALGO = "ppo"
SEEDS = [17, 18, 19]
TIMESTEPS = 800000

times = np.zeros(len(SEEDS))

for i, seed in enumerate(SEEDS):

    env = gym.make("Oekolopoly-v1")
    env = OekoBoxActionWrapper(env)
    env = OekoPerRoundRewardWrapper(env, per_round_reward=0.5)
    env = Monitor(env, f"timing/{ALGO}_{seed}")
    env.reset()

    if ALGO == "ppo":
        model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    elif ALGO == "sac":
        model = SAC("MlpPolicy", env, verbose=0, seed=seed)
    elif ALGO == "td3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=0.1 * np.ones(n_actions))
        model = TD3("MlpPolicy", env, verbose=0, seed=seed,
                    action_noise=action_noise)

    t0 = time.time()
    model.learn(total_timesteps=TIMESTEPS, log_interval=1, progress_bar=True)
    exec_time = time.time() - t0
    print(exec_time)
    times[i] = exec_time

    Path(f"timing/{ALGO}_{seed}").mkdir(parents=True)
    model.save(f"timing/{ALGO}_{seed}/end_of_training_agent")
    l = env.get_episode_lengths()
    rews = env.get_episode_rewards()
    total_steps = env.get_total_steps()
    np.savez_compressed(f"timing/{ALGO}_{seed}/train_hist.npz",
                        training_lengths=l,
                        training_rewards=rews,
                        training_timesteps=total_steps)

    del env
    del model

with open(f"timing/{ALGO}.npy", 'wb') as f:
    np.save(f, times)

print(f"Training time: {np.mean(times)} +/- {np.std(times)}")
