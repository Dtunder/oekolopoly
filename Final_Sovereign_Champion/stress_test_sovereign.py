import os
import sys
import numpy as np
import gymnasium as gym
import torch
from sb3_contrib import RecurrentPPO

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from run_champion import SovereignGuardian

def run_stress_test(num_episodes=50, noise_level=2):
    print(f"--- STARTING SOVEREIGN STRESS TEST ---")
    print(f"Noise Level: +/- {noise_level} units on initial state")
    
    model_path = os.path.join(ROOT, "sota_recurrent_champion.zip")
    model = RecurrentPPO.load(model_path, device='cpu')
    base_env = gym.make("Oekolopoly-v2")
    guardian = SovereignGuardian(base_env)
    
    results = []
    with torch.inference_mode():
        for i in range(num_episodes):
            obs, _ = base_env.reset()
            # Inject Noise into initial state
            V = base_env.unwrapped.V
            for j in range(8): # Noise on San, Pro, Edu, QoL, PopG, Env, Pop, Pol
                noise = np.random.randint(-noise_level, noise_level + 1)
                V[j] = np.clip(V[j] + noise, base_env.unwrapped.Vmin[j], base_env.unwrapped.Vmax[j])
            
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            year = 0
            for _ in range(40):
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                final_action = guardian.get_final_action(action, int(base_env.unwrapped.V[9]))
                obs, reward, terminated, truncated, info = base_env.step(final_action)
                episode_starts = np.zeros((1,), dtype=bool)
                year = base_env.unwrapped.V[8]
                if terminated or truncated:
                    break
            results.append(year)
            if (i+1) % 10 == 0:
                print(f"Episode {i+1}/{num_episodes} completed. Survival Rate: {sum(1 for r in results if r >= 30)/(i+1)*100:.1f}%")
    
    print("\n--- STRESS TEST RESULTS ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Success Rate (30 Years) under Noise: {sum(1 for r in results if r >= 30)/num_episodes*100:.1f}%")
    if sum(1 for r in results if r >= 30) == num_episodes:
        print("STATUS: ABSOLUTE ROBUSTNESS CONFIRMED.")
    else:
        print("STATUS: VULNERABILITY DETECTED UNDER EXTREME INITIAL CONDITIONS.")

if __name__ == "__main__":
    run_stress_test()
