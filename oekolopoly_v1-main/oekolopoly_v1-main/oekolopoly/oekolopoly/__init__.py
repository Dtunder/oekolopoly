import gym
from gym.envs.registration import register

#if 'Oekolopoly-v1' in gym.envs.registration.registry.env_specs:
#    del gym.envs.registration.registry.env_specs['Oekolopoly-v1']

register(
    id='Oekolopoly-v1',
    entry_point='oekolopoly.oekolopoly.envs:OekoEnv',
)
