import os
import sys
import time
import logging
import numpy as np
import gymnasium as gym

import io
# Force UTF-8 for Windows terminal to support Nasuta's cool ASCII graphics
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- 1. THE TRANSPARENT NEURAL FIX (Monkey Patch) ---
# We fix the PyTorch/Numpy 2.0 'int64' mismatch here so the Neural Network can work.
def apply_neural_fix():
    import torch.nn as nn
    original_lstm_init = nn.LSTM.__init__
    def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
        # We force input_size and hidden_size to be pure Python ints.
        return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
    nn.LSTM.__init__ = patched_lstm_init
    print("[SYSTEM] Neural Logic Fixed: LSTM now accepts current Environment shapes.")

apply_neural_fix()

# --- 2. PATHS & INFRASTRUCTURE ---
ROOT = os.path.dirname(os.path.abspath(__file__))
NASUTA_ROOT = r"G:\Meine Ablage\oekolopoly2\gymcts-games-main\gymcts-games-main\src"
if NASUTA_ROOT not in sys.path:
    sys.path.insert(0, NASUTA_ROOT)

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_action_history_wrapper import ActionHistoryMCTSGymEnvWrapper
from oekolopoly.env.oeko_env import OekoEnv, OekoActionBuilderWrapper
import oekolopoly.env.oeko_env as oeko_env
sys.modules['oekolopoly.oekolopoly'] = oeko_env

# --- 3. THE SOVEREIGN SYSTEMATIC AGENT ---
class SystematicSovereign:
    def __init__(self, model_path):
        from sb3_contrib import RecurrentPPO
        print(f"[SYSTEM] Loading Sovereign Model: {model_path}")
        self.model = RecurrentPPO.load(model_path, device='cpu')
        
    def action_mask_fn(self, env):
        # We search for the ActionBuilder in the wrapper stack
        curr = env
        while hasattr(curr, 'env'):
            if hasattr(curr, 'valid_action_mask'):
                return curr.valid_action_mask()
            curr = curr.env
        # Fallback to the method we know exists in OekoActionBuilderWrapper
        return env.unwrapped.valid_action_mask()

    def run_full_game(self, render=True):
        # 1. Setup Env (The Systematic Nasuta Way)
        base_env = OekoEnv(render_mode="ansi")
        wrapped_env = OekoActionBuilderWrapper(base_env, auxilary_reward=True)
        env = ActionHistoryMCTSGymEnvWrapper(wrapped_env, action_mask_fn=self.action_mask_fn)
        
        # --- API BRIDGES (MUST BE BEFORE AGENT INIT) ---
        def is_terminal_bridge():
            return env.unwrapped.done
        def get_valid_actions_bridge():
            mask = self.action_mask_fn(env)
            # Find current AP in stack
            curr = env
            ap = 0
            while hasattr(curr, 'env'):
                if hasattr(curr, '_available_action_points'):
                    ap = curr._available_action_points
                    break
                curr = curr.env
            # Force investment
            if ap > 0: mask[0] = False
            valid_ids = [i for i, m in enumerate(mask) if m]
            return valid_ids if valid_ids else [0]
            
        def sovereign_rollout():
            import torch
            obs = env.unwrapped.obs
            obs_fixed = np.array([obs], dtype=np.float32)
            lstm_states = (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256)) 
            episode_starts = torch.ones(1, dtype=torch.float32)
            value = self.model.policy.predict_values(torch.as_tensor(obs_fixed), lstm_states, episode_starts)
            return float(value[0][0]) + 5.0

        env.is_terminal = is_terminal_bridge
        env.get_valid_actions = get_valid_actions_bridge
        env.rollout = sovereign_rollout

        # --- IMPORTANT: RESET BEFORE AGENT INIT ---
        # This initializes the 'V' vector in the environment
        obs, _ = env.reset()

        # 2. create the agent AFTER environment is initialized
        agent = GymctsAgent(
            env=env,
            clear_mcts_tree_after_step=False,
            number_of_simulations_per_step=1000,
            render_tree_after_step=False
        )
        
        done = False
        round_count = 0
        
        print("\n" + "="*60)
        print("   SOVEREIGN SYSTEMATIC ENGINE: GAME START")
        print("="*60)

        while not done:
            round_count += 1
            if render: print(env.env.env.render())
            
            print(f"\n[Thinking...] Round {round_count} Strategy Calculation...")
            step_count = 0
            while True:
                # 3. SYSTEMATIC STEP
                best_action, next_node = agent.perform_mcts_step(num_simulations=1000)
                
                print(f"  [MCTS] Action Selected: {best_action} (N={next_node.visit_count}, Q={next_node.mean_value:.2f})")
                agent.show_mcts_tree(tree_max_depth=1)

                # Execute in REAL environment
                obs, reward, terminated, truncated, info = env.step(best_action)
                done = terminated or truncated
                step_count += 1

                # Safe AP Check for Log
                curr_ap = 0
                c = env
                while hasattr(c, 'env'):
                    if hasattr(c, '_available_action_points'):
                        curr_ap = c._available_action_points
                        break
                    c = c.env

                # Action 0 means round ended
                if best_action == 0 or done:
                    print(f"Round {round_count} completed in {step_count} strategic steps.")
                    break
                else:
                    print(f"  > Spent 1 AP on Action {best_action}. (Remaining: {curr_ap})")

            if done:
                print("\n" + "="*60)
                print("   GAME OVER")
                print(f"   Reason: {info.get('done_reason', 'Unknown')}")
                print(f"   Final Score (Balance): {info.get('balance', 0)}")
                print("="*60)
                break

if __name__ == "__main__":
    MODEL_PATH = os.path.join(ROOT, "sota_recurrent_champion.zip")
    engine = SystematicSovereign(MODEL_PATH)
    engine.run_full_game()
