import sys
import os
import time
import gymnasium as gym
import numpy as np
import io

# Force UTF-8 for Windows terminal to support Nasuta's cool ASCII graphics
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure paths
ROOT = os.path.dirname(os.path.abspath(__file__))
# Priority paths for Nasuta's original work
NASUTA_ROOT = r"G:\Meine Ablage\oekolopoly2\gymcts-games-main\gymcts-games-main\src"
if NASUTA_ROOT not in sys.path:
    sys.path.insert(0, NASUTA_ROOT)

import oekolopoly.env.oeko_env as oeko_env
from oekolopoly.env.oeko_env import OekoEnv

# Manual Gym Registration
from gymnasium.envs.registration import register
try:
    register(
        id='Oekolopoly-v2',
        entry_point='oekolopoly.env.oeko_env:OekoEnv',
    )
except:
    pass

# Import bridge for mcts_planner
import sys
sys.modules['oekolopoly.oekolopoly'] = oeko_env
from mcts_planner import SovereignMCTS
from run_champion import SovereignGuardian
from sb3_contrib import RecurrentPPO

def watch_nasuta():
    print("====================================================")
    print("   WATCHING NASUTA'S CHAMPION (ORIGINAL MODE)   ")
    print("====================================================")
    
    # 1. Init Nasuta's specific Env with Wrapper
    from oekolopoly.env.oeko_env import OekoActionBuilderWrapper
    base_env = OekoEnv()
    env = OekoActionBuilderWrapper(base_env)
    
    # 2. Load Model
    model_path = os.path.join(ROOT, "sota_recurrent_champion.zip")
    model = RecurrentPPO.load(model_path, device='cpu')
    
    # 3. Create Planner
    planner = SovereignMCTS(model, num_simulations=100) # Balanced thinking
    
    obs, _ = env.reset()
    done = False
    step_num = 0
    
    while not done:
        step_num += 1
        # Render the board (ASCII) - Use unwrapped for render if needed
        print(env.env.render()) # Nasuta's original render
        
        round_display = int(env.env.unwrapped.V[8]) + 1
        print(f"\n[Sovereign Champion is thinking... (Action {step_num}, Round {round_display})]")
        
        # Get optimal action from MCTS
        # action is discrete (0-8)
        action = planner.search(env)
        
        print(f"Strategic Action Selected: {action}")
        
        # Step in the wrapped environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        time.sleep(0.5) 
        
        if terminated or truncated:
            print(env.env.render())
            print("\n--- SIMULATION TERMINATED ---")
            print(f"Reason: {info.get('done_reason')}")
            break

if __name__ == "__main__":
    watch_nasuta()
