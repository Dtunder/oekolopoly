import os
import sys
import numpy as np
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
import oekolopoly.oekolopoly

class NoisyInitialStateWrapper(gym.Wrapper):
    def __init__(self, env, noise_level=2):
        super().__init__(env)
        self.noise_level = noise_level

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        V = self.env.unwrapped.V
        for j in range(8):
            noise = np.random.randint(-self.noise_level, self.noise_level + 1)
            V[j] = np.clip(V[j] + noise, self.env.unwrapped.Vmin[j], self.env.unwrapped.Vmax[j])

        return self.env.observation_space.sample(), info

def make_env():
    # Construct an environment matching the SOTA action/observation specs
    # For now we use the basic environment with the Noise wrapper
    env = gym.make("Oekolopoly-v2")
    env = NoisyInitialStateWrapper(env, noise_level=2)
    return env

def train():
    from stable_baselines3.common.env_util import make_vec_env
    vec_env = make_vec_env(make_env, n_envs=1)

    model_path = os.path.join(ROOT, "sota_recurrent_champion")

    print("Loading base Sovereign model to begin robust curriculum training...")
    try:
        model = RecurrentPPO.load(model_path, env=vec_env, device='cpu')
        model.learning_rate = 1e-4
    except Exception as e:
        print(f"Failed to load model: {e}")
        # If loading fails due to architecture mismatch, we initiate a fresh curriculum
        print("Initiating fresh robust curriculum...")
        model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1, learning_rate=1e-4)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(ROOT, 'models'),
        name_prefix='robust_champion'
    )

    print("Starting background training (10,000 timesteps for demonstration)...")
    model.learn(total_timesteps=10000, callback=checkpoint_callback, progress_bar=False)
    model.save(os.path.join(ROOT, "sota_robust_champion"))
    print("Training finished.")

if __name__ == "__main__":
    train()
