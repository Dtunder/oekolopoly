import sys
import os
import time

# Priority paths for Nasuta's original work ONLY
NASUTA_ROOT = r"G:\Meine Ablage\oekolopoly2\gymcts-games-main\gymcts-games-main\src"
if NASUTA_ROOT not in sys.path:
    sys.path.insert(0, NASUTA_ROOT)

from oekolopoly.env.oeko_env import OekoEnv, OekoActionBuilderWrapper
from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper

def run_pure_nasuta():
    print("====================================================")
    print("   PURE NASUTA RESEARCH LAB - NO SOVEREIGN ADDONS  ")
    print("====================================================")
    print("Using original MCTS GymctsAgent and Konen's Engine.\n")
    
    # 1. Init Original Env with Nasuta's Action Builder & DeepCopy
    env = OekoEnv()
    env = OekoActionBuilderWrapper(env)
    env = DeepCopyMCTSGymEnvWrapper(env)
    
    # MCTS Bridges (Robust recursive search for Nasuta compatibility)
    def find_valid_mask():
        curr = env
        while curr is not None:
            if hasattr(curr, 'valid_action_mask'):
                return curr.valid_action_mask()
            curr = getattr(curr, 'env', None)
        return [True] * 9 # Fallback
        
    env.get_valid_actions = lambda: [i for i, valid in enumerate(find_valid_mask()) if valid]
    
    obs, _ = env.reset()
    
    # 2. Init Original MCTS Agent
    # number_of_simulations_per_step is for EACH point allocation
    agent = GymctsAgent(env=env, number_of_simulations_per_step=50) 
    
    print("Simulation started. Watching every round (1 to 30)...")
    
    while not env.unwrapped.done:
        # Nasuta's agent decides on a single action (spending 1 point or finishing round)
        action, _ = agent.perform_mcts_step(num_simulations=50)
        
        # Apply to environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # If action is 0, the round (year) is finished in the ActionBuilder logic
        if action == 0:
            v = env.unwrapped.V
            round_idx = int(v[8])
            print(f"\n[ROUND {round_idx} COMPLETED]")
            print(f" > Environment: {int(v[5])}")
            print(f" > Quality of Life: {int(v[3])}")
            print(f" > Population: {int(v[6])}")
            print(f" > Budget for Next Round: {int(v[9])}")
            
            if round_idx >= 30:
                break
        
        if terminated or truncated:
            break

    print("\n====================================================")
    print(f"EXPERIMENT FINISHED at Round {int(env.unwrapped.V[8])}")
    print(f"Final Balance: {env.unwrapped.balance:.2f}")
    print(f"Reason: {env.unwrapped.done_info}")
    print("====================================================")

if __name__ == "__main__":
    run_pure_nasuta()
