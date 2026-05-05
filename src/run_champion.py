import os
import sys
import gc
import logging
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Optional, Any

# Configure Professional Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SovereignRunner")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch

# Priority paths for Nasuta's original work
NASUTA_ROOT = r"G:\Meine Ablage\oekolopoly2\gymcts-games-main\gymcts-games-main\src"
if NASUTA_ROOT not in sys.path:
    sys.path.insert(0, NASUTA_ROOT)

torch.set_num_threads(1)
torch.set_grad_enabled(False)

# Manual Environment Registration
from gymnasium.envs.registration import register
try:
    register(
        id="Oekolopoly-v2",
        entry_point="oekolopoly.env.oeko_env:OekoEnv",
    )
except Exception:
    pass # Already registered

# WATERPROOF MONKEY PATCH
import torch.nn as nn

# Sovereign Component Imports
from mcts_planner import SovereignMCTS
from sb3_contrib import RecurrentPPO
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init

# Set paths dynamically
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

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

def run_sovereign() -> None:
    """Executes a full 30-round simulation using the Sovereign Champion."""
    logger.info("--- SOVEREIGN CHAMPION V130: MASTER ALCHEMIST ---")
    gc.collect()
    
    try:
        model_path = os.path.join(ROOT, "sota_recurrent_champion.zip")
        model = RecurrentPPO.load(model_path, device='cpu')
        base_env = gym.make("Oekolopoly-v2")
        planner = SovereignMCTS(model, num_simulations=3000, render_tree=True)
    
        obs, _ = base_env.reset()
        V = base_env.unwrapped.V
        
        for round_num in range(1, 35):
            # DEEP THINKING SEARCH
            final_action = planner.search(base_env)
            
            obs, reward, terminated, truncated, info = base_env.step(final_action)
            V = base_env.unwrapped.V
            
            logger.info(f"Round {int(V[8])}: Env={int(V[5])}, QoL={int(V[3])}, AP={int(V[9])}")
                
            if terminated or truncated:
                if V[8] >= 30:
                    logger.info(f"SUCCESS: Round 30 reached. Final AP: {int(V[9])}")
                else:
                    logger.warning(f"FAILED at Round {int(V[8])}. Reason: {info.get('done_reason')}")
                break
                
    except Exception as e:
        logger.error(f"Critical error during simulation: {e}")

if __name__ == "__main__":
    run_sovereign()
