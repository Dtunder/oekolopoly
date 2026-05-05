import os
import sys

ROOT_DIR = r"G:\Meine Ablage\oekolopoly2"
SRC_DIR = os.path.join(ROOT_DIR, "src")
NASUTA_ROOT = os.path.join(ROOT_DIR, r"gymcts-games-main\gymcts-games-main\src")

if NASUTA_ROOT not in sys.path:
    sys.path.insert(0, NASUTA_ROOT)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

print("Paths set.")

try:
    import gymnasium as gym
    print("Gymnasium imported.")
    import torch
    print("Torch imported.")
    from sb3_contrib import RecurrentPPO
    print("RecurrentPPO imported.")
except Exception as e:
    print(f"Import Error: {e}")

try:
    import oekolopoly.env.oeko_env as oeko_env
    print(f"oekolopoly.env.oeko_env imported from {oeko_env.__file__}")
except Exception as e:
    print(f"oekolopoly.env.oeko_env Import Error: {e}")

try:
    import oekolopoly.oekolopoly.envs.oeko_env as oeko_env_v2
    print(f"oekolopoly.oekolopoly.envs.oeko_env imported from {oeko_env_v2.__file__}")
except Exception as e:
    print(f"oekolopoly.oekolopoly.envs.oeko_env Import Error: {e}")
