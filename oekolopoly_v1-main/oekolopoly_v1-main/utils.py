"""Utility functions and classes for Oekolopoly environment"""

from stable_baselines3.common.monitor import Monitor

def decode_from_agent_string(agent_str):
    """Gets parameters from directory name in /agents """
    t = agent_str.split("_")
    obs = t[1]
    act = t[3]
    rew = t[5]
    shape = float(t[6])
    drl = t[-2]
    seed = t[-1]

    return obs, act, rew, shape, drl, seed

class EvalMonitor(Monitor):
    """Runs one evaluation episode after completion of training episode"""
    def __init__(self, env, eval_env, filename=None, info_keywords=None):
        super().__init__(env, filename, info_keywords=info_keywords)
        self.eval_env = eval_env

    def set_model(self, model):
        self.model = model

    def step(self, action):
        s, r, done, info = super().step(action)
        if done:
            eval_s = self.eval_env.reset()
            eval_done = False
            while not eval_done:
                eval_a, _ = self.model.predict(eval_s, deterministic=True)
                eval_s, _, eval_done, _ = self.eval_env.step(eval_a)

        return s, r, done, info
