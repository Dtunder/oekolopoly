import os
import sys
import numpy as np
import torch
import torch.nn as nn
# MONKEY PATCH for PyTorch 2.x compatibility
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, *args, **kwargs):
    if isinstance(input_size, np.int64): input_size = int(input_size)
    return original_lstm_init(self, input_size, *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init
from sb3_contrib import RecurrentPPO

# Add paths for C: Drive
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "oekolopoly"))

# Import core Nasuta components
from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_action_history_wrapper import ActionHistoryMCTSGymEnvWrapper
from oekolopoly.oekolopoly.envs.oeko_env import OekoEnv
from wrappers import OekoActionBuilderWrapper, HomeostaticRewardV3

class SystematicSovereign:
    def __init__(self, model_path):
        print(f"[SYSTEM] Loading Sovereign Model: {model_path}")
        self.model = RecurrentPPO.load(model_path)

    def action_mask_fn(self, env):
        # Traverse stack for mask
        curr = env
        while hasattr(curr, 'env'):
            if hasattr(curr, 'valid_action_mask'):
                return curr.valid_action_mask()
            curr = curr.env
        return np.ones(9, dtype=bool)

    def run_full_game(self):
        # 1. Setup Env with SURVIVAL LOGIC
        base_env = OekoEnv(render_mode="ansi")
        
        # Inject Homeostatic Survival Reward
        env = HomeostaticRewardV3(base_env)
        
        # Nasuta Sequential Building Wrapper
        wrapped_env = OekoActionBuilderWrapper(env)
        
        # Action History for Deterministic MCTS
        env = ActionHistoryMCTSGymEnvWrapper(wrapped_env, action_mask_fn=self.action_mask_fn)
        
        # --- API BRIDGES ---
        def is_terminal_bridge(): return env.unwrapped.done
        def get_valid_actions_bridge():
            mask = self.action_mask_fn(env)
            # Find current AP
            curr = env
            ap = 0
            while hasattr(curr, 'env'):
                if hasattr(curr, '_available_action_points'):
                    ap = curr._available_action_points
                    break
                curr = curr.env
            # FORCE INVESTMENT (End Round mask)
            if ap > 0: mask[0] = False
            valid_ids = [i for i, m in enumerate(mask) if m]
            return valid_ids if valid_ids else [0]
            
        def sovereign_rollout():
            # Find the ActionBuilder to get current round investments
            curr = env
            edu_bonus = 0.0
            pol_bonus = 0.0
            while hasattr(curr, 'env'):
                if hasattr(curr, '_current_action_dict'):
                    # Target Education 21
                    v = env.unwrapped.V
                    edu_total = v[2] + curr._current_action_dict["Education"]
                    if edu_total < 21:
                        edu_bonus = curr._current_action_dict["Education"] * 0.4
                    
                    # Target Politics 15
                    pol_val = v[7]
                    if pol_val < 15:
                        # Sanitation and QoL help Politics
                        pol_bonus = (curr._current_action_dict["Sanitation"] + curr._current_action_dict["Quality of Life"]) * 0.5
                    break
                curr = curr.env
                
            obs = env.unwrapped.obs
            obs_fixed = np.array([obs], dtype=np.float32)
            lstm_states = (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256)) 
            episode_starts = torch.ones(1, dtype=torch.float32)
            val = self.model.policy.predict_values(torch.as_tensor(obs_fixed), lstm_states, episode_starts).detach()
            return float(val[0][0]) + edu_bonus + pol_bonus

        env.is_terminal = is_terminal_bridge
        env.get_valid_actions = get_valid_actions_bridge
        env.rollout = sovereign_rollout
        env.reset()

        # 2. Setup Agent
        agent = GymctsAgent(
            env=env,
            clear_mcts_tree_after_step=True,
            render_tree_after_step=True,
            number_of_simulations_per_step=1000,
            render_tree_max_depth=1,
            exclude_unvisited_nodes_from_render=True
        )
        
        print("\n" + "="*60)
        print("   SOVEREIGN CHAMPION: SURVIVAL MODE (30 ROUND TARGET)")
        print("==================================================")
        
        # 3. SOLVE (Until Death or 30 Rounds)
        try:
            agent.solve()
            final_info = env.unwrapped.info
            print("\n" + "="*60)
            print("   SIMULATION CONCLUDED")
            print(f"   Reason: {final_info.get('done_reason', 'Unknown')}")
            print(f"   Detail: {final_info.get('done_reason_detail', 'N/A')}")
            print(f"   Final Year: {env.unwrapped.V[8]}")
            print("="*60)
        except Exception as e:
            print(f"\n[SYSTEM] Simulation error: {str(e)}")

if __name__ == "__main__":
    MODEL_PATH = os.path.join(ROOT, "sota_recurrent_champion.zip")
    engine = SystematicSovereign(MODEL_PATH)
    engine.run_full_game()
