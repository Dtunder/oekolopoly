import sys
import os
sys.path.append(os.path.abspath('Final_Sovereign_Champion'))

import gymnasium as gym
from sb3_contrib import RecurrentPPO
import oekolopoly.oekolopoly
from oekolopoly.wrappers import OekoActionBuilderWrapper, HomeostaticRewardV3

env = gym.make("Oekolopoly-v2")
env = OekoActionBuilderWrapper(env)
env = HomeostaticRewardV3(env)
print("Environment Observation Space:", env.observation_space)

model = RecurrentPPO.load("Final_Sovereign_Champion/sota_recurrent_champion.zip", device="cpu")
print("Model Observation Space:", model.observation_space)
