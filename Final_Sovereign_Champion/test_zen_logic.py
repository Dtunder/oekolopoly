import os
import sys
import numpy as np
import gymnasium as gym

# Set paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

import oekolopoly.oekolopoly
from run_champion import SovereignGuardian

def test_zen():
    print("Initializing Zen Logic Test...")
    env = gym.make("Oekolopoly-v2")
    guardian = SovereignGuardian(env)
    
    obs, _ = env.reset()
    done = False
    year = 0
    
    print("Year | Env | QoL | Pol | AP | Reason")
    print("-" * 50)
    
    while not done and year < 35:
        V = env.unwrapped.V
        avail = int(V[9])
        
        # We simulate a "dummy" raw action since we don't use the model
        raw_action = np.zeros(6) 
        
        final_action = guardian.get_final_action(raw_action, avail)
        obs, reward, terminated, truncated, info = env.step(final_action)
        done = terminated or truncated
        year = int(env.unwrapped.V[8])
        
        V_new = env.unwrapped.V
        print(f"{year:4} | {int(V_new[5]):3} | {int(V_new[3]):3} | {int(V_new[7]):3} | {int(V_new[9]):2} | {info.get('done_reason', 'OK')}")
        
    if year >= 30:
        print("\nSUCCESS: Year 30 reached with Zen Logic!")
    else:
        print(f"\nFAILED at Year {year}. Reason: {info.get('done_reason')}")

if __name__ == "__main__":
    test_zen()
