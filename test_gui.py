import sys
import os
import argparse
sys.path.append(os.path.abspath('Final_Sovereign_Champion'))
os.chdir(os.path.abspath('Final_Sovereign_Champion'))

import gymnasium as gym
import oekolopoly.oekolopoly
from oekolopoly.wrappers import OekoActionBuilderWrapper
env = gym.make("Oekolopoly-v2")

env.reset()

import oekolopoly_gui
guardian = oekolopoly_gui.SovereignGuardian(env)
avail = int(env.unwrapped.V[env.unwrapped.POINTS])
action, reason = guardian.get_final_action(avail)
print("Guardian Action:", action)
print("Type of Action:", type(action))

# Also testing predict_future
future = guardian.predict_future(action)
print("Future:", future)
