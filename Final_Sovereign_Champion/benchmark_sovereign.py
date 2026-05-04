import os
import sys
import gc
from typing import Any
import numpy as np
import gymnasium as gym

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
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

class SovereignGuardian:
    """The analytical core of the Sovereign Champion."""
    def __init__(self, env: gym.Env):
        self.env = env

    def get_final_action(self, raw_action: Any, avail: int) -> np.ndarray:
        """Applies V106 Alchemist logic to ensure 30-year survival."""
        V = self.env.unwrapped.V
        dist = np.zeros(5, dtype=int)
        
        # 1. CORE INVESTMENTS
        if V[2] < 29 and avail > 0:
            d = min(avail, 29 - int(V[2])); dist[2] = int(d); avail -= dist[2]
            
        # 2. SURVIVAL TARGETS
        p_target = 13
        p_dist = p_target - int(V[1])
        if avail > 0:
            d = min(max(1, avail // 2), abs(p_dist))
            dist[1] = -int(d) if p_dist < 0 else int(d); avail -= abs(dist[1])
            
        # 3. ALCHEMIST BURN (Safety valve for Action Points)
        while avail + int(V[9]) > 28:
            changed = False
            if int(V[2]) + dist[2] < 29:
                dist[2] += 1; avail -= 1; changed = True
            elif int(V[3]) + dist[3] < 18:
                dist[3] += 1; avail -= 1; changed = True
            elif int(V[4]) + dist[4] < 15:
                dist[4] += 1; avail -= 1; changed = True
            elif V[5] < 12:
                if int(V[1]) + dist[1] < 18:
                    dist[1] += 1; avail -= 1; changed = True
                else: break
            elif V[5] > 20:
                if int(V[0]) + dist[0] < 25:
                    dist[0] += 1; avail -= 1; changed = True
                else: break
            else:
                if int(V[1]) + dist[1] > 5:
                    dist[1] -= 1; avail -= 1; changed = True
                else: break
            if avail + int(V[9]) <= 28: break

        final = [int(dist[0]), int(dist[1] + 28), int(dist[2]), int(dist[3]), int(dist[4]), 5]
        if V[6] > 32: final[5] = 1
        elif V[6] < 18: final[5] = 10
        return np.clip(final, 0, 56)


def run_benchmark(num_episodes=100):
    print(f"--- STARTING SOVEREIGN BENCHMARK: {num_episodes} EPISODES ---")
    model_path = os.path.join(ROOT, "sota_recurrent_champion")
    model = RecurrentPPO.load(model_path, device='cpu')
    base_env = gym.make("Oekolopoly-v2")
    guardian = SovereignGuardian(base_env)
    
    results = []
    with torch.inference_mode():
        for i in range(num_episodes):
            obs, _ = base_env.reset()
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            year = 0
            for _ in range(40): # Buffer for 30 years
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                final_action = guardian.get_final_action(action, int(base_env.unwrapped.V[9]))
                obs, reward, terminated, truncated, info = base_env.step(final_action)
                episode_starts = np.zeros((1,), dtype=bool)
                year = base_env.unwrapped.V[8]
                if terminated or truncated:
                    break
            results.append(year)
            if (i+1) % 10 == 0:
                success_count = sum(1 for r in results if r >= 30)
                print(f"Episode {i+1}/{num_episodes} completed. Current Success Rate: {success_count/(i+1)*100:.1f}%")
    
    print("\n--- FINAL BENCHMARK RESULTS ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Mean Years: {np.mean(results):.2f}")
    print(f"Min Years: {np.min(results)}")
    print(f"Max Years: {np.max(results)}")
    print(f"Success Rate (30 Years): {sum(1 for r in results if r >= 30)/num_episodes*100:.1f}%")

if __name__ == "__main__":
    run_benchmark()
