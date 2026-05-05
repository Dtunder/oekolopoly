import sys
import gymnasium as gym
import numpy as np
import copy

sys.path.append('Final_Sovereign_Champion')
import oekolopoly.oekolopoly
from mcts_planner import MCTS
from benchmark_sovereign import SovereignGuardian, ROOT, RecurrentPPO
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
import torch.nn as nn
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init

model_path = os.path.join(ROOT, "sota_recurrent_champion.zip")
model = RecurrentPPO.load(model_path, device='cpu')
base_env = gym.make("Oekolopoly-v2")
guardian = SovereignGuardian(base_env)
obs, _ = base_env.reset()
lstm_states = None
ep_start = True
for year in range(8):
    print(f"--- YEAR {year} ---")
    print("V Before:", base_env.unwrapped.V)
    planner = MCTS(base_env, model, guardian, num_simulations=10)
    best_action, pv, next_lstm = planner.search(base_env, root_lstm_states=lstm_states, episode_start=ep_start)
    print("Best action selected:", best_action)
    lstm_states = next_lstm
    ep_start = False

    obs, reward, terminated, truncated, info = base_env.step(np.array(best_action, dtype=np.int64))
    print("V After:", base_env.unwrapped.V)
    print("Info:", info)
    if terminated or truncated:
        print("Done!")
        break
