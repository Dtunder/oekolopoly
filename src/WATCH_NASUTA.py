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
    
    # 1. Init Nasuta's specific Env
    env = OekoEnv()
    
    # 2. Load Model
    model_path = os.path.join(ROOT, "sota_recurrent_champion.zip")
    model = RecurrentPPO.load(model_path, device='cpu')
    
    # 3. Create Planner
    planner = SovereignMCTS(model, num_simulations=50) # Fast thinking for watching
    guardian = SovereignGuardian(env)
    
    obs, _ = env.reset()
    
    for round_num in range(35):
        # Clear screen for animation effect
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Render the board (ASCII)
        print(env.render())
        
        print(f"\n[Nasuta is thinking about Round {round_num + 1}...]")
        
        # Get optimal action
        action = planner.search(env)
        avail = int(env.unwrapped.V[9])
        final_action = guardian.get_final_action(action, avail)
        
        print(f"Action Selection: {final_action}")
        
        # Step
        obs, reward, terminated, truncated, info = env.step(final_action)
        
        time.sleep(1) # Wait a second so you can watch
        
        if terminated or truncated:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(env.render())
            print("\n--- SIMULATION TERMINATED ---")
            print(f"Reason: {info.get('done_reason')}")
            break

if __name__ == "__main__":
    watch_nasuta()
