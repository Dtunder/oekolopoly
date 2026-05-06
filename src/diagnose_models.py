import os
import glob
import traceback
import gymnasium as gym
from sb3_contrib import RecurrentPPO

# Paths
ROOT_DIR = r"G:\Meine Ablage\oekolopoly2"
SRC_DIR = os.path.join(ROOT_DIR, "src")
NASUTA_ROOT = os.path.join(ROOT_DIR, r"gymcts-games-main\gymcts-games-main\src")

import sys
if NASUTA_ROOT not in sys.path:
    sys.path.insert(0, NASUTA_ROOT)

import torch.nn as nn
import gymnasium as gym
sys.modules['gym'] = gym

# WATERPROOF MONKEY PATCH for LSTM (Fixes Numpy 2.0 int64 issues)
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init

import oekolopoly.env.oeko_env as oeko_env
from oekolopoly.env.oeko_env import OekoEnv, OekoActionBuilderWrapper
sys.modules['oekolopoly.oekolopoly'] = oeko_env

def run_diagnostics():
    print("=" * 60)
    print("   SOVEREIGN MODEL DIAGNOSTICS & COMPATIBILITY SCAN")
    print("=" * 60)
    
    # Setup test environment
    try:
        base_env = OekoEnv()
        test_env = OekoActionBuilderWrapper(base_env)
        obs, _ = test_env.reset()
        obs_shape = obs.shape
        print(f"[ENV] Current Observation Space Shape: {obs_shape}")
    except Exception as e:
        print(f"[ENV ERROR] Failed to init environment: {e}")
        return

    # Find all zip files
    model_files = glob.glob(os.path.join(ROOT_DIR, "**", "*.zip"), recursive=True)
    
    print(f"\nFound {len(model_files)} compressed model artifacts. Commencing scan...\n")
    
    report = {"working": [], "broken": []}
    
    for mf in model_files:
        model_name = os.path.basename(mf)
        folder_name = os.path.basename(os.path.dirname(mf))
        print(f"Testing: {folder_name} / {model_name}")
        
        try:
            # Try loading
            model = RecurrentPPO.load(mf, device='cpu')
            
            # Try inference (check if observation space matches)
            action, _states = model.predict(obs, deterministic=True)
            
            print("  [✓] STATUS: OPERATIONAL. Model is fully compatible with current Env.")
            report["working"].append(mf)
            
        except Exception as e:
            err_msg = str(e).split('\n')[0] # Get first line of error
            print(f"  [X] STATUS: INCOMPATIBLE.")
            print(f"      Reason: {err_msg}")
            
            # Diagnose common reasons
            if "size mismatch" in err_msg.lower() or "shape" in err_msg.lower():
                print("      Diagnosis: Observation Space architecture mismatch. This model was trained on an older version of the environment.")
            
            report["broken"].append((mf, err_msg))
            
        print("-" * 40)
        
    print("\n" + "=" * 60)
    print("   DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Total Models Scanned: {len(model_files)}")
    print(f"Fully Compatible:     {len(report['working'])}")
    print(f"Broken/Incompatible:  {len(report['broken'])}")
    
if __name__ == "__main__":
    run_diagnostics()
