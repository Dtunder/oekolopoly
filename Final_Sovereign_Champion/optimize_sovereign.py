import os
import sys
import numpy as np
import gymnasium as gym
import optuna
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# WATERPROOF MONKEY PATCH
import torch.nn as nn
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
import oekolopoly.oekolopoly
from sb3_contrib import RecurrentPPO
from run_champion import SovereignGuardian

def objective(trial):
    edu_max = trial.suggest_int('edu_max', 20, 29)
    p_target = trial.suggest_int('p_target', 10, 20)
    qol_burn = trial.suggest_int('qol_burn', 15, 25)
    pop_burn = trial.suggest_int('pop_burn', 10, 20)
    env_burn_low = trial.suggest_int('env_burn_low', 5, 15)
    env_burn_high = trial.suggest_int('env_burn_high', 16, 25)

    model_path = os.path.join(ROOT, "sota_recurrent_champion.zip")
    try:
        model = RecurrentPPO.load(model_path, device='cpu')
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0

    base_env = gym.make("Oekolopoly-v2")
    guardian = SovereignGuardian(base_env, edu_max=edu_max, p_target=p_target,
                                 qol_burn=qol_burn, pop_burn=pop_burn,
                                 env_burn_low=env_burn_low, env_burn_high=env_burn_high)

    num_episodes = 5
    results = []

    with torch.inference_mode():
        for _ in range(num_episodes):
            obs, _ = base_env.reset()
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            year = 0
            for _ in range(30):
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                final_action = guardian.get_final_action(action, int(base_env.unwrapped.V[9]))
                obs, reward, terminated, truncated, info = base_env.step(final_action)
                episode_starts = np.zeros((1,), dtype=bool)
                year = base_env.unwrapped.V[8]
                if terminated or truncated:
                    break
            results.append(year)

    return np.mean(results)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
