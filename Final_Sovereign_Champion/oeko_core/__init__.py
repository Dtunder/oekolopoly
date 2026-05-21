import gymnasium as gym
from gymnasium.envs.registration import register

#if 'oeko_core-v1' in gym.envs.registration.registry.env_specs:
#    del gym.envs.registration.registry.env_specs['oeko_core-v1']

register(
    id='oeko_core-v2',
    entry_point='oeko_core.envs:OekoEnv',
)
