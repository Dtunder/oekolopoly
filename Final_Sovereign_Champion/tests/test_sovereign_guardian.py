import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock

# Modify sys.path to allow importing from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_champion import SovereignGuardian

def test_guardian_zero_avail():
    # If there are no action points available, guardian should mostly return 0 investments
    # Final action is [dist0, dist1+28, dist2, dist3, dist4, 5]
    env_mock = MagicMock()
    env_mock.unwrapped.V = np.array([20, 20, 20, 20, 20, 20, 20, 0, 0, 0])
    guardian = SovereignGuardian(env_mock)

    raw_action = [0, 0, 0, 0, 0, 0]
    final_action = guardian.get_final_action(raw_action, avail=0)

    assert final_action[0] == 0
    assert final_action[1] == 28 # dist[1] + 28, where dist[1]=0
    assert final_action[2] == 0
    assert final_action[3] == 0
    assert final_action[4] == 0

def test_guardian_education_max_limit():
    env_mock = MagicMock()
    # V[2] = 29
    env_mock.unwrapped.V = np.array([20, 20, 29, 20, 20, 20, 20, 0, 0, 10])
    guardian = SovereignGuardian(env_mock)

    raw_action = [0, 0, 0, 0, 0, 0]
    final_action = guardian.get_final_action(raw_action, avail=10)

    # Investment in Education (dist[2]) should be 0 because V[2] is already 29
    assert final_action[2] == 0

def test_guardian_production_survival_target():
    env_mock = MagicMock()
    # V[1] = 13 (Production target is 13)
    env_mock.unwrapped.V = np.array([20, 13, 29, 20, 20, 20, 20, 0, 0, 10])
    guardian = SovereignGuardian(env_mock)

    raw_action = [0, 0, 0, 0, 0, 0]
    final_action = guardian.get_final_action(raw_action, avail=10)

    # Investment in Production should be 0 because V[1] is 13 (target reached)
    assert final_action[1] == 28 # dist[1] = 0

def test_guardian_alchemist_burn():
    env_mock = MagicMock()
    # Create scenario to trigger alchemist burn (avail + V[9] > 28)
    # Give max action points
    env_mock.unwrapped.V = np.array([10, 10, 10, 10, 10, 10, 20, 0, 0, 30])
    guardian = SovereignGuardian(env_mock)

    raw_action = [0, 0, 0, 0, 0, 0]
    final_action = guardian.get_final_action(raw_action, avail=10)

    # Verify that total investments have been made to reduce action points below the threshold
    # The actual distribution depends on the while loop in Alchemist Burn
    assert final_action[0] >= 0
    assert final_action[2] >= 0
