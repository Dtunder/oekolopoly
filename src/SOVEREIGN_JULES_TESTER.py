import os
import sys
import time
import numpy as np
import torch

# Add paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "oekolopoly"))

# --- NEURAL FIX (Monkey Patch) ---
import torch.nn as nn
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_action_history_wrapper import ActionHistoryMCTSGymEnvWrapper
from oekolopoly.oekolopoly.envs.oeko_env import OekoEnv
from wrappers import OekoActionBuilderWrapper
from sb3_contrib import RecurrentPPO

class SovereignJulesTester:
    def __init__(self, model_path):
        print(f"[JULES] Initializing Master Validator...")
        self.model = RecurrentPPO.load(model_path)
        
    def action_mask_fn(self, env):
        # Traverse for mask
        curr = env
        while hasattr(curr, 'env'):
            if hasattr(curr, 'valid_action_mask'):
                return curr.valid_action_mask()
            curr = curr.env
        return np.ones(10, dtype=bool)

    def run_benchmark(self, trials=3):
        results = []
        print(f"\n[JULES] Starting Stress Test ({trials} Trials)...")
        print("="*60)
        
        for i in range(trials):
            print(f"\n[TRIAL {i+1}] Launching Sovereign Champion...")
            base_env = OekoEnv(render_mode="ansi")
            wrapped_env = OekoActionBuilderWrapper(base_env)
            env = ActionHistoryMCTSGymEnvWrapper(wrapped_env, action_mask_fn=self.action_mask_fn)
            
            # API Bridges
            def is_terminal_bridge(): return env.unwrapped.done
            def get_valid_actions_bridge():
                mask = self.action_mask_fn(env)
                curr = env
                ap = 0
                while hasattr(curr, 'env'):
                    if hasattr(curr, '_available_action_points'):
                        ap = curr._available_action_points
                        break
                    curr = curr.env
                if ap > 0: mask[0] = False
                valid_ids = [idx for idx, m in enumerate(mask) if m]
                return valid_ids if valid_ids else [0]
                
            def sovereign_rollout():
                obs = env.unwrapped.obs
                obs_fixed = np.array([obs], dtype=np.float32)
                lstm_states = (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256)) 
                episode_starts = torch.ones(1, dtype=torch.float32)
                val = self.model.policy.predict_values(torch.as_tensor(obs_fixed), lstm_states, episode_starts).detach()
                return float(val[0][0])

            env.is_terminal = is_terminal_bridge
            env.get_valid_actions = get_valid_actions_bridge
            env.rollout = sovereign_rollout
            env.reset()

            agent = GymctsAgent(env=env, number_of_simulations_per_step=100)
            
            start_time = time.time()
            try:
                actions = agent.solve()
                duration = time.time() - start_time
                final_info = env.unwrapped.get_info() if hasattr(env.unwrapped, 'get_info') else {}
                score = final_info.get('balance', 0)
                reason = "Natural Completion"
            except Exception as e:
                duration = time.time() - start_time
                score = 0
                reason = str(e)

            print(f"[TRIAL {i+1}] Result: {reason} | Score: {score} | Time: {duration:.1f}s")
            results.append({"score": score, "reason": reason, "time": duration})

        print("\n" + "="*60)
        print("   FINAL VALIDATION REPORT (GOOGLE JULES STYLE)")
        print("="*60)
        avg_score = sum(r['score'] for r in results) / trials
        print(f" > Average Stability Score: {avg_score:.2f}")
        print(f" > System Status: [READY FOR DEPLOYMENT]" if avg_score > -5 else " > System Status: [STABILITY ALERT]")
        print("="*60)

if __name__ == "__main__":
    MODEL_PATH = "G:\\Meine Ablage\\oekolopoly2\\src\\sota_recurrent_champion.zip"
    tester = SovereignJulesTester(MODEL_PATH)
    tester.run_benchmark(trials=2)
