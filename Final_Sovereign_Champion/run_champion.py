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

# Set paths dynamically
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

import oekolopoly.oekolopoly

def lazy_load_torch():
    """Bypasses hangs on Py3.14 by loading torch only when needed."""
    try:
        import torch
        import torch.nn as nn
        from sb3_contrib import RecurrentPPO
        
        # WATERPROOF MONKEY PATCH
        original_lstm_init = nn.LSTM.__init__
        def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
            return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
        nn.LSTM.__init__ = patched_lstm_init
        
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        return torch, RecurrentPPO
    except ImportError:
        logger.error("Torch or SB3 not found. AI features disabled.")
        return None, None

class SovereignGuardian:
    """The analytical core of the Sovereign Champion."""
    def __init__(self, env: gym.Env):
        self.env = env

    def get_final_action(self, raw_action: Any, avail: int) -> Tuple[np.ndarray, List[str]]:
        """Applies V290 Alchemist logic with reasoning logs."""
        V = self.env.unwrapped.V
        dist = np.zeros(5, dtype=int)
        reasons = []
        
        # 1. CORE INVESTMENTS
        if V[2] < 29 and avail > 0:
            d = min(avail, 29 - int(V[2]))
            dist[2] = int(d)
            avail -= dist[2]
            reasons.append(f"Edu+{dist[2]}")
            
        # 2. SURVIVAL TARGETS & BLACK SKY
        p_target = 13
        if V[5] <= 9 or V[7] < -5:
            p_target = 22 # Force high production (Black Sky Shield)
            reasons.append("BLACK SKY ACTIVE")
        else:
            reasons.append("Zen Stability")

        p_dist = p_target - int(V[1])
        if avail > 0:
            d = min(avail, abs(p_dist)) if (V[5] <= 9 or V[7] < -5) else min(max(1, avail // 2), abs(p_dist))
            dist[1] = -int(d) if p_dist < 0 else int(d)
            avail -= abs(dist[1])
            if dist[1] != 0: reasons.append(f"Prod {'+' if dist[1]>0 else ''}{dist[1]}")
            
        # 3. ALCHEMIST BURN (Safety valve for Action Points)
        if avail + int(V[9]) > 28:
            reasons.append("Alchemist Burn")
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
                if not changed or avail + int(V[9]) <= 28: break

        final = [int(dist[0]), int(dist[1] + 28), int(dist[2]), int(dist[3]), int(dist[4]), 5]
        if V[6] > 32: final[5] = 1
        elif V[6] < 18: final[5] = 10
        
        return np.clip(final, 0, 56), reasons

def run_sovereign() -> None:
    """Executes a full 30-year simulation using the Sovereign Champion."""
    logger.info("--- SOVEREIGN CHAMPION V130: MASTER ALCHEMIST ---")
    gc.collect()
    
    try:
        model_path = os.path.join(ROOT, "sota_recurrent_champion.zip")
        model = RecurrentPPO.load(model_path, device='cpu')
        base_env = gym.make("Oekolopoly-v2")
        guardian = SovereignGuardian(base_env)
        
        obs, _ = base_env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        for year in range(35):
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            final_action, reasons = guardian.get_final_action(action, int(base_env.unwrapped.V[9]))
            obs, reward, terminated, truncated, info = base_env.step(final_action)
            episode_starts = np.zeros((1,), dtype=bool)
            V = base_env.unwrapped.V
            
            if year % 5 == 0:
                logger.info(f"Year {int(V[8])}: Env={int(V[5])}, QoL={int(V[3])}, AP={int(V[9])} | Reasons: {', '.join(reasons)}")
                
            if terminated or truncated:
                break
                
        if V[8] >= 30:
            logger.info(f"SUCCESS: Year 30 reached. Final AP: {int(V[9])}")
        else:
            logger.warning(f"FAILED at Year {int(V[8])}. Reason: {info.get('done_reason')}")
            
    except Exception as e:
        logger.error(f"Critical error during simulation: {e}")

if __name__ == "__main__":
    run_sovereign()
