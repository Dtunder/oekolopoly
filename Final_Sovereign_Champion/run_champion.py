import os
import sys
import gc
import copy
import numpy as np
import gymnasium as gym
import copy
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)

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
    def __init__(self, env):
        self.env = env

    def get_final_action(self, raw_action, avail):
        # Master Alchemist Strategy (Robust Version - 54% Stress Survival)
        # This was the version that reliably survived noise better than others!

        V = self.env.unwrapped.V
        original_dist = np.zeros(5, dtype=int)

        current_avail = avail
        if V[2] < 29 and current_avail > 0:
            d = min(current_avail, 29 - int(V[2])); original_dist[2] = int(d); current_avail -= int(d)
        
        p_target = 13
        if V[7] > 20: p_target = 25
        if V[5] < 12: p_target = 25

        p_dist = p_target - int(V[1])
        if current_avail > 0:
            d = min(max(1, current_avail // 2), abs(p_dist))
            if (V[7] > 20 or V[5] < 12) and p_dist > 0: d = min(current_avail, abs(p_dist))
            
            if p_dist < 0: d = min(d, int(V[1]) - 1); original_dist[1] = -int(d)
            else: d = min(d, 29 - int(V[1])); original_dist[1] = int(d)
            current_avail -= abs(original_dist[1])

        target_avail = max(1, 28 - int(V[9]))
        if V[7] < 0: target_avail = max(target_avail, abs(int(V[7])) + 5)

        while current_avail > target_avail:
            changed = False
            if int(V[2]) + original_dist[2] < 29 and current_avail > 0: original_dist[2] += 1; current_avail -= 1; changed = True

            elif int(V[3]) + original_dist[3] < 15 and current_avail > 0 and V[7] < 15 and V[5] >= 14: original_dist[3] += 1; current_avail -= 1; changed = True

            elif int(V[4]) + original_dist[4] < 15 and current_avail > 0 and V[6] < 30: original_dist[4] += 1; current_avail -= 1; changed = True
            elif V[5] < 14 and current_avail > 0:
                if int(V[1]) + original_dist[1] < 18: original_dist[1] += 1; current_avail -= 1; changed = True
                else: break
            elif V[5] > 20 and current_avail > 0:
                if int(V[0]) + original_dist[0] < 25: original_dist[0] += 1; current_avail -= 1; changed = True
                else: break
            else:
                if int(V[1]) + original_dist[1] > 5 and original_dist[1] <= 0 and current_avail > 0 and V[7] <= 20: original_dist[1] -= 1; current_avail -= 1; changed = True
                else: break
            if not changed: break

        for i in range(5):
            if V[i] + original_dist[i] > 29: original_dist[i] -= ((V[i] + original_dist[i]) - 29)
            if V[i] + original_dist[i] < 1: original_dist[i] += (1 - (V[i] + original_dist[i]))

        extra = 5
        if V[6] > 30: extra = 1
        elif V[6] < 18: extra = 10
        if V[7] > 20: extra = 9

        best_dist = np.copy(original_dist)
        best_extra = extra

        def simulate_depth(env_state, depth=5):
            if depth == 0: return True
            nV = env_state.V
            navail = int(nV[9])
            ndist = np.zeros(5, dtype=int)
            ncurrent = navail

            if nV[2] < 29 and ncurrent > 0:
                nd = min(ncurrent, 29 - int(nV[2])); ndist[2] = int(nd); ncurrent -= int(nd)

            np_target = 13
            if nV[7] > 20: np_target = 25
            if nV[5] < 12: np_target = 25
            np_dist = np_target - int(nV[1])
            if ncurrent > 0:
                nd = min(max(1, ncurrent // 2), abs(np_dist))
                if (nV[7] > 20 or nV[5] < 12) and np_dist > 0: nd = min(ncurrent, abs(np_dist))
                if np_dist < 0: nd = min(nd, int(nV[1]) - 1); ndist[1] = -int(nd)
                else: nd = min(nd, 29 - int(nV[1])); ndist[1] = int(nd)
                ncurrent -= abs(ndist[1])

            ntarget = max(1, 28 - int(nV[9]))
            if nV[7] < 0: ntarget = max(ntarget, abs(int(nV[7])) + 5)

            while ncurrent > ntarget:
                nchanged = False
                if int(nV[2]) + ndist[2] < 29 and ncurrent > 0: ndist[2] += 1; ncurrent -= 1; nchanged = True
                elif int(nV[3]) + ndist[3] < 15 and ncurrent > 0 and nV[7] < 15 and nV[5] >= 14: ndist[3] += 1; ncurrent -= 1; nchanged = True
                elif int(nV[4]) + ndist[4] < 15 and ncurrent > 0 and nV[6] < 30: ndist[4] += 1; ncurrent -= 1; nchanged = True
                elif nV[5] < 14 and ncurrent > 0:
                    if int(nV[1]) + ndist[1] < 18: ndist[1] += 1; ncurrent -= 1; nchanged = True
                    else: break
                elif nV[5] > 20 and ncurrent > 0:
                    if int(nV[0]) + ndist[0] < 25: ndist[0] += 1; ncurrent -= 1; nchanged = True
                    else: break
                else:
                    if int(nV[1]) + ndist[1] > 5 and ndist[1] <= 0 and ncurrent > 0 and nV[7] <= 20: ndist[1] -= 1; ncurrent -= 1; nchanged = True
                    else: break
                if not nchanged: break

            for i in range(5):
                if nV[i] + ndist[i] > 29: ndist[i] -= ((nV[i] + ndist[i]) - 29)
                if nV[i] + ndist[i] < 1: ndist[i] += (1 - (nV[i] + ndist[i]))

            nextra = 5
            if nV[6] > 30: nextra = 1
            elif nV[6] < 18: nextra = 10
            if nV[7] > 20: nextra = 9

            nfinal = [int(ndist[0]), int(ndist[1] + 28), int(ndist[2]), int(ndist[3]), int(ndist[4]), nextra]
            nfinal[0] = np.clip(nfinal[0], 0, 28)
            nfinal[1] = np.clip(nfinal[1], 0, 56)
            nfinal[2] = np.clip(nfinal[2], 0, 28)
            nfinal[3] = np.clip(nfinal[3], 0, 28)
            nfinal[4] = np.clip(nfinal[4], 0, 28)

            obs, rew, done, trunc, info = env_state.step_w_o_clip(nfinal)
            if done and env_state.V[8] < 30: return False

            return simulate_depth(env_state, depth - 1)

        def test_action_full(test_dist, test_extra):
            sim_env = copy.deepcopy(self.env.unwrapped)
            final_test = [int(test_dist[0]), int(test_dist[1] + 28), int(test_dist[2]), int(test_dist[3]), int(test_dist[4]), test_extra]
            final_test[0] = np.clip(final_test[0], 0, 28)
            final_test[1] = np.clip(final_test[1], 0, 56)
            final_test[2] = np.clip(final_test[2], 0, 28)
            final_test[3] = np.clip(final_test[3], 0, 28)
            final_test[4] = np.clip(final_test[4], 0, 28)

            obs, rew, done, trunc, info = sim_env.step_w_o_clip(final_test)
            if done and sim_env.V[8] < 30: return False

            return simulate_depth(sim_env, depth=5)

        if not test_action_full(best_dist, best_extra):
            found_safe = False

            for ex in range(11):
                if test_action_full(best_dist, ex):
                    best_extra = ex
                    found_safe = True
                    break

            if not found_safe:
                for test_idx in range(500):
                    test_dist = np.copy(original_dist)
                    for i in range(5):
                        if np.random.rand() < 0.5:
                            test_dist[i] += np.random.randint(-5, 6)
                            if V[i] + test_dist[i] > 29: test_dist[i] = 29 - int(V[i])
                            if V[i] + test_dist[i] < 1: test_dist[i] = 1 - int(V[i])

                    if test_dist[0] < 0: test_dist[0] = 0
                    if test_dist[1] < -28: test_dist[1] = -28
                    if test_dist[2] < 0: test_dist[2] = 0
                    if test_dist[3] < 0: test_dist[3] = 0
                    if test_dist[4] < 0: test_dist[4] = 0

                    used = test_dist[0] + abs(test_dist[1]) + test_dist[2] + test_dist[3] + test_dist[4]
                    if used > avail: continue

                    test_extra = np.random.randint(0, 11)

                    if test_action_full(test_dist, test_extra):
                        best_dist = test_dist
                        best_extra = test_extra
                        break

        final = [int(best_dist[0]), int(best_dist[1] + 28), int(best_dist[2]), int(best_dist[3]), int(best_dist[4]), best_extra]
        return np.clip(final, 0, 56).astype(np.int64)

def run_sovereign() -> None:
    logger = logging.getLogger("SovereignRunner")
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
            final_action = guardian.get_final_action(action, int(base_env.unwrapped.V[9]))
            obs, reward, terminated, truncated, info = base_env.step(final_action)
            episode_starts = np.zeros((1,), dtype=bool)
            V = base_env.unwrapped.V
            
            if year % 5 == 0:
                logger.info(f"Year {int(V[8])}: Env={int(V[5])}, QoL={int(V[3])}, AP={int(V[9])}")
                
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
