"""
    Parse arguments as in train.py, load the appropriate agent from file
    and run one episode with this agent
"""

import argparse
from pathlib import Path
import numpy as np
import gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor

import oekolopoly.oekolopoly        # this import is needed, otherwise gym.make will not succeed
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


def load_model(env, savedir, args):
    """
    load a trained Oekolopoly agent from file

    :param env:     environment, needed for observation_space and action_space
    :param savedir: directory of agent
    :param args:    needed for args.algo (type of SB3 model)
    :return:        the model (trained SB3 agent)
    """
    agent_file = f"{savedir}.zip"
    if Path(agent_file).is_file():
        if args.algo == "ppo":
            model = PPO.load(agent_file,
                             custom_objects={'observation_space': env.observation_space,
                                             'action_space': env.action_space})
        elif args.algo == "sac":
            model = SAC.load(agent_file,
                             custom_objects={'observation_space': env.observation_space,
                                             'action_space': env.action_space})
        elif args.algo == "td3":
            model = TD3.load(agent_file,
                             custom_objects={'observation_space': env.observation_space,
                                             'action_space': env.action_space})
        else:
            raise ValueError("Unavailable DRL algoritm")
    else:
        raise ValueError(f"Could not find agent file: {agent_file}")

    return model


def transf_act_box(env, act):
    """
    transform the box action points in the same way as ``OekoBoxActionWrapper.action(self,act)`` does

    :param env: environment of type ``OekoBoxActionWrapper``
    :param act: the box action points
    :return: the points properly transformed and clipped for the Oekolopoly wheels
    """
    act = np.clip(act, env.action_min, env.action_max, dtype=np.float32)
    assert env.action_space.contains(act), "Action not in action_space"

    if act[1] < 0:
        act[1] = -act[1]
        reduce_production = True
    else:
        reduce_production = False

    regions_act = act[0:5]
    special_act = round(act[5] * 5)
    regions_act = env.distribute1(regions_act, env.V[env.POINTS])
    if reduce_production: regions_act[1] = -regions_act[1]

    for i in range(len(regions_act)):
        region_result = env.V[i] + regions_act[i]
        if   region_result < env.Vmin[i]: regions_act[i] = env.Vmin[i] - env.V[i]
        elif region_result > env.Vmax[i]: regions_act[i] = env.Vmax[i] - env.V[i]

    act = np.append(regions_act, special_act)

    return act


def transf_act_simple(env, act):
    """
    transform the simple action points in the same way as ``OekoSimpleActionWrapper.action(self,act)`` does

    :param env: environment of type ``OekoSimpleActionWrapper``
    :param act: the simple action points
    :return: the points properly transformed and clipped for the Oekolopoly wheels
    """
    action_index = act[0]
    extra_points = act[1]

    points = env.V[env.POINTS]
    act_string = env.ACTIONS[action_index]
    regions = [0, 0, 0, 0, 0]

    remaining = 0
    for i in range(5):
        region_points_float = points / 3 * int(act_string[i])
        region_points_int = np.floor(region_points_float)
        remaining += region_points_float - region_points_int
        regions[i] = region_points_int
    remaining = round(remaining)

    if remaining:
        for i in range(5):
            if int(act_string[i]) > 0:
                regions[i] += 1
                remaining -= 1
                if remaining == 0: break
    assert remaining == 0

    if int(act_string[5]) == 1:
        regions[1] = -regions[1]

    for i in range(5):
        region_result = env.V[i] + regions[i]
        if   region_result < env.Vmin[i]:  regions[i] = env.Vmin[i] - env.V[i]
        elif region_result > env.Vmax[i]:  regions[i] = env.Vmax[i] - env.V[i]

    act = np.int32(np.append(regions, extra_points))

    return act


if __name__ == "__main__":
    args = parse_args()

    agent_str = "obs_" + args.observation + "_action_" + args.action + "_reward_" + args.reward + "_" + str(args.shaping) + "_" + args.algo + "_" + str(args.seed)
    savedir = f"./trained agents/{agent_str}"

    # Prepare environment
    env = gym.make("Oekolopoly-v1")
    env = wrap_env(env, args, savedir+"/train", monitor=False)

    # Load model
    model = load_model(env, savedir, args)

    # Run episode
    s = env.reset()
    m_return = 0
    done = False
    while not done:
        action, _ = model.predict(s, deterministic=True)
        if args.action == "box":
            points = transf_act_box(env, action)     # the points as they are distributed to the wheels
            print(s, points)
        elif args.action == "simple":
            points = transf_act_simple(env, action)  # the points as they are distributed to the wheels
            print(s, points, action)
        else:
            print(s, action)
        s, r, done, info = env.step(action)
        m_return += r

    print(s)
    print(f"Episode finished with return={m_return},\n info={info}")
