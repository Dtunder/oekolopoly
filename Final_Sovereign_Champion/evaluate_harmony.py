import os
import sys
import numpy as np
import gymnasium as gym
import torch
import copy
from sb3_contrib import RecurrentPPO

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from benchmark_sovereign import SovereignGuardian

def calculate_harmony():
    print("--- SOVEREIGN HARMONY EVALUATOR ---")
    
    model_path = os.path.join(ROOT, "sota_recurrent_champion.zip")
    model = RecurrentPPO.load(model_path, device='cpu')
    base_env = gym.make("Oekolopoly-v2")
    guardian = SovereignGuardian(base_env)
    
    obs, _ = base_env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    history = []
    for _ in range(35):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        final_action = guardian.get_final_action(action, int(base_env.unwrapped.V[9]))
        obs, reward, terminated, truncated, info = base_env.step(final_action)
        episode_starts = np.zeros((1,), dtype=bool)
        history.append(copy.deepcopy(base_env.unwrapped.V))
        if terminated or truncated: break
    
    history = np.array(history)
    years = len(history)
    
    # METRICS
    avg_qol = np.mean(history[:, 3])
    avg_env = np.mean(history[:, 5])
    avg_edu = np.mean(history[:, 2])
    avg_pop = np.mean(history[:, 6])
    
    # SCORING (Lower is better for penalty)
    # QoL should be near 15
    qol_score = abs(avg_qol - 15)
    # Env should be near 15
    env_score = abs(avg_env - 15)
    # Edu should be 29
    edu_score = 29 - avg_edu
    # Pop should be near 25
    pop_score = abs(avg_pop - 25)
    
    total_penalty = qol_score + env_score + edu_score + pop_score
    
    grade = "F"
    if years >= 30:
        if total_penalty < 5: grade = "A+"
        elif total_penalty < 10: grade = "A"
        elif total_penalty < 15: grade = "B"
        elif total_penalty < 20: grade = "C"
        else: grade = "D"
    
    print(f"\nSimulation Results (Year {years}):")
    print(f"Average QoL: {avg_qol:.2f} (Target: 15)")
    print(f"Average Env: {avg_env:.2f} (Target: 15)")
    print(f"Average Edu: {avg_edu:.2f} (Target: 29)")
    print(f"Average Pop: {avg_pop:.2f} (Target: 25)")
    print("-" * 30)
    print(f"TOTAL HARMONY PENALTY: {total_penalty:.2f}")
    print(f"FINAL GRADE: {grade}")
    print("-" * 30)

if __name__ == "__main__":
    calculate_harmony()
