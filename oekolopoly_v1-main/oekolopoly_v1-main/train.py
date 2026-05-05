import json
import argparse
from pathlib import Path
import numpy as np
import gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from utils import EvalMonitor
import oekolopoly.oekolopoly
from oekolopoly.wrappers import OekoSimpleObservationWrapper, OekoBoxObservationWrapper, OekoSimpleActionWrapper, OekoBoxActionWrapper, OekoPerRoundRewardWrapper, OekoAuxRewardWrapper, OekoBoxUnclippedActionWrapper


def parse_args():
    """
    Parse arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Training parameters for Oekolopoly')
    parser.add_argument('--seed', type=int,
                        help='Seed for reproducible results')
    parser.add_argument('--observation', type=str, default="none",
                        help='Observation wrapper (box, simple or none)')
    parser.add_argument('--action', type=str, default="none",
                        help='Action wrapper (box, simple, unclippedbox or none)')
    parser.add_argument('--reward', type=str, default="none",
                        help='Reward wrapper (perround, aux or none)')
    parser.add_argument('--shaping', type=float, default=1.,
                        help='Parameter for reward shaping')
    parser.add_argument('--timesteps', type=int, default=int(8e5),
                        help='Number of training timesteps for each trial')
    parser.add_argument('--algo', type=str, default="ppo", choices=['ppo', 'sac', 'td3'],
                        help='DRL algorithm')

    args = parser.parse_args()

    return args

def wrap_env(env, args, savedir, monitor=False):

    # Wrap observations
    if args.observation == "simple":
        env = OekoSimpleObservationWrapper(env)
    elif args.observation == "box":
        env = OekoBoxObservationWrapper(env)
    elif args.observation == "none":
        pass
    else:
        raise ValueError("No valid string for observation wrapper passed")

    # Wrap actions
    if args.action == "simple":
        env = OekoSimpleActionWrapper(env)
    elif args.action == "box":
        env = OekoBoxActionWrapper(env)
    elif args.action == "unclippedbox":
        env = OekoBoxUnclippedActionWrapper(env)
    elif args.action == "none":
        pass
    else:
        raise ValueError("No valid string for action wrapper passed")

    # Wrap rewards
    if args.reward == "perround":
        env = OekoPerRoundRewardWrapper(env, per_round_reward=args.shaping)
    elif args.reward == "aux":
        env = OekoAuxRewardWrapper(env, scaling=args.shaping)
    elif args.reward == "none":
        pass
    else:
        raise ValueError("No valid string for reward wrapper passed")

    if monitor:
        env = Monitor(env, savedir, info_keywords=("balance (always)",
                                                   "balance_numerator (always)",
                                                   "balance", "balance_numerator",
                                                   "round",
                                                   "done_reason",
                                                   "valid_move",
                                                   "invalid_move_info"))

    return env

if __name__ == "__main__":
    args = parse_args()

    json_object = json.dumps(args.__dict__, indent=4)

    agent_str = "obs_" + args.observation + "_action_" + args.action + "_reward_" + args.reward + "_" + str(args.shaping) + "_" + args.algo + "_" + str(args.seed)
    savedir = f"./agents/{agent_str}"

    Path(savedir).mkdir(parents=True)
    with open(f"{savedir}/training_parameters.json", "w") as outfile:
        outfile.write(json_object)

    # Prepare evaluation environment
    eval_env = gym.make("Oekolopoly-v1")
    eval_env = wrap_env(eval_env, args, savedir+"/eval", monitor=True)
    eval_env.reset()

    # Prepare training environment and wrap in EvalMonitor
    env = gym.make("Oekolopoly-v1")
    env = wrap_env(env, args, savedir+"/train", monitor=False)
    env = EvalMonitor(env, eval_env, savedir+"/train", info_keywords=("balance (always)", "balance_numerator (always)", "balance", "balance_numerator", "round", "done_reason", "valid_move", "invalid_move_info"))
    env.reset()

    # Set up model
    if args.algo == "ppo":
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    elif args.algo == "sac":
        model = SAC("MlpPolicy", env, verbose=1, seed=args.seed)
    elif args.algo == "td3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3("MlpPolicy", env, verbose=1, seed=args.seed, action_noise=action_noise)
    else:
        raise ValueError("Unavailable DRL algoritm")
    env.set_model(model)  # Make model available for prediction during evaluation

    cust_logger = configure(f"{savedir}", ["csv", "log"])
    model.set_logger(cust_logger)
    model.learn(total_timesteps=args.timesteps, log_interval=1, progress_bar=True)
    model.save(f"{savedir}/end_of_training_agent")
    l = env.get_episode_lengths()
    rews = env.get_episode_rewards()
    total_steps = env.get_total_steps()
    np.savez_compressed(f"{savedir}/train_hist.npz", training_lengths=l, training_rewards=rews, training_timesteps=total_steps)
