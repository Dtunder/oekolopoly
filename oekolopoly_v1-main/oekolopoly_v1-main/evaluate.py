"""Computes average and standard deviation of last WINDOW evaluation
episodes and compiles dataset "results_eot_mean{WINDOW}"
This should be fed into `plot_last_training_episodes.py`
"""

from pathlib import Path
import pandas as pd

import agent_lists
from utils import decode_from_agent_string


if __name__ == "__main__":

    WINDOW = 1000
    AGENTS = agent_lists.PPO_WK

    obs_list = []
    action_list = []
    rew_list = []
    shape_list = []
    drl_list = []
    seed_list = []

    rounds_mean_list = []
    rounds_std_list = []
    balance_mean_list = []
    balance_std_list = []

    for agent_str in AGENTS:
        print(agent_str)
        obs, action, rew, shape, drl, seed = decode_from_agent_string(agent_str)

        if Path(f"./agents/{agent_str}/eval.monitor.csv").is_file():

            df = pd.read_csv(f"./agents/{agent_str}/eval.monitor.csv", header=1)
            rounds_mean_list.append(df["round"].iloc[-WINDOW:].mean())
            rounds_std_list.append(df["round"].iloc[-WINDOW:].std())

            balance_mean_list.append(df["balance"].iloc[-WINDOW:].mean())
            balance_std_list.append(df["balance"].iloc[-WINDOW:].std())

            obs_list.append(obs)
            action_list.append(action)
            rew_list.append(rew)
            shape_list.append(shape)
            drl_list.append(drl)
            seed_list.append(seed)

    results = pd.DataFrame({"DRL": drl_list,
                            "obs_wrapper": obs_list,
                            "action_wrapper": action_list,
                            "reward_wrapper": rew_list,
                            "shape": shape_list,
                            "seed": seed_list,
                            "rounds": rounds_mean_list,
                            "rounds std": rounds_std_list,
                            "balance": balance_mean_list,
                            "balance std": balance_std_list})

    print(results)
    assert len(AGENTS) == len(results)
    results.to_csv(f"results_{WINDOW}_fluc")
