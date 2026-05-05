import os
import sys
import gc
import logging
import numpy as np
import gymnasium as gym
import copy
import optuna
from typing import Any

# Ensure determinism and prevent PyTorch memory leaks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# WATERPROOF MONKEY PATCH for LSTM
import torch.nn as nn
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init

# Set paths dynamically
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

import oekolopoly.oekolopoly
from sb3_contrib import RecurrentPPO

class TunableSovereignGuardian:
    """A tunable version of the Sovereign Guardian for Optuna optimization."""
    def __init__(self, env: gym.Env, params: dict):
        self.env = env
        self.params = params

    def get_final_action(self, raw_action: Any, avail: int) -> np.ndarray:
        """Applies parameterized logic to ensure 30-year survival."""
        V = self.env.unwrapped.V
        dist = np.zeros(5, dtype=np.int32)

        # 1. CORE INVESTMENTS (Education Target)
        edu_target = self.params['edu_target']
        if V[2] < edu_target and avail > 0:
            d = min(avail, edu_target - int(V[2])); dist[2] = int(d); avail -= dist[2]

        # 2. SURVIVAL TARGETS (Production)
        p_target = self.params['p_target']
        p_dist = p_target - int(V[1])
        if avail > 0:
            d = min(max(1, avail // 2), abs(p_dist))
            dist[1] = -int(d) if p_dist < 0 else int(d); avail -= abs(dist[1])

        # 3. ALCHEMIST BURN (Safety valve for Action Points)
        ap_burn_threshold = self.params['ap_burn_threshold']
        edu_burn_target = self.params['edu_burn_target']
        qol_burn_target = self.params['qol_burn_target']
        pop_growth_burn_target = self.params['pop_growth_burn_target']
        env_low_threshold = self.params['env_low_threshold']
        prod_burn_target_low_env = self.params['prod_burn_target_low_env']
        env_high_threshold = self.params['env_high_threshold']
        san_burn_target_high_env = self.params['san_burn_target_high_env']
        prod_burn_target_default = self.params['prod_burn_target_default']

        while avail + int(V[9]) > ap_burn_threshold:
            changed = False
            if int(V[2]) + dist[2] < edu_burn_target:
                dist[2] += 1; avail -= 1; changed = True
            elif int(V[3]) + dist[3] < qol_burn_target:
                dist[3] += 1; avail -= 1; changed = True
            elif int(V[4]) + dist[4] < pop_growth_burn_target:
                dist[4] += 1; avail -= 1; changed = True
            elif V[5] < env_low_threshold:
                if int(V[1]) + dist[1] < prod_burn_target_low_env:
                    dist[1] += 1; avail -= 1; changed = True
                else: break
            elif V[5] > env_high_threshold:
                if int(V[0]) + dist[0] < san_burn_target_high_env:
                    dist[0] += 1; avail -= 1; changed = True
                else: break
            else:
                if int(V[1]) + dist[1] > prod_burn_target_default:
                    dist[1] -= 1; avail -= 1; changed = True
                else: break
            if avail + int(V[9]) <= ap_burn_threshold: break

        final = [int(dist[0]), int(dist[1] + 28), int(dist[2]), int(dist[3]), int(dist[4]), 5]

        pop_high_threshold = self.params['pop_high_threshold']
        pop_low_threshold = self.params['pop_low_threshold']
        if V[6] > pop_high_threshold: final[5] = 1
        elif V[6] < pop_low_threshold: final[5] = 10

        return np.clip(final, 0, 56)


def objective(trial):
    # Hyperparameters based on V290 logic mapping
    params = {
        'edu_target': trial.suggest_int('edu_target', 20, 29),
        'p_target': trial.suggest_int('p_target', 8, 18),
        'ap_burn_threshold': trial.suggest_int('ap_burn_threshold', 25, 30),
        'edu_burn_target': trial.suggest_int('edu_burn_target', 20, 29),
        'qol_burn_target': trial.suggest_int('qol_burn_target', 12, 22),
        'pop_growth_burn_target': trial.suggest_int('pop_growth_burn_target', 10, 20),
        'env_low_threshold': trial.suggest_int('env_low_threshold', 8, 15),
        'prod_burn_target_low_env': trial.suggest_int('prod_burn_target_low_env', 15, 25),
        'env_high_threshold': trial.suggest_int('env_high_threshold', 16, 25),
        'san_burn_target_high_env': trial.suggest_int('san_burn_target_high_env', 20, 29),
        'prod_burn_target_default': trial.suggest_int('prod_burn_target_default', 0, 10),
        'pop_high_threshold': trial.suggest_int('pop_high_threshold', 25, 40),
        'pop_low_threshold': trial.suggest_int('pop_low_threshold', 10, 22),
    }

    model_path = os.path.join(ROOT, "sota_recurrent_champion.zip")
    model = RecurrentPPO.load(model_path, device='cpu')
    base_env = gym.make("Oekolopoly-v2")
    guardian = TunableSovereignGuardian(base_env, params)

    obs, _ = base_env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    history = []
    for _ in range(35):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        final_action = guardian.get_final_action(action, int(base_env.unwrapped.V[9]))
        obs, reward, terminated, truncated, info = base_env.step(final_action)
        episode_starts = np.zeros((1,), dtype=bool)
        history.append(copy.deepcopy(base_env.unwrapped.V))
        if terminated or truncated: break

    history = np.array(history)
    years = len(history)

    if years < 30:
        # Massive penalty for dying to heavily encourage survival
        return 1000.0 + (30 - years) * 100.0

    # Harmony Metric (Lower is better)
    avg_qol = np.mean(history[:, 3])
    avg_env = np.mean(history[:, 5])
    avg_edu = np.mean(history[:, 2])
    avg_pop = np.mean(history[:, 6])

    qol_score = abs(avg_qol - 15)
    env_score = abs(avg_env - 15)
    edu_score = 29 - avg_edu
    pop_score = abs(avg_pop - 25)

    total_penalty = qol_score + env_score + edu_score + pop_score

    return total_penalty


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    # We use a deterministic sampler for mathematical proof
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)

    print("\n--- OPTUNA OPTIMIZATION FINISHED ---")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (Penalty): {study.best_value}")
    print("Best params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
