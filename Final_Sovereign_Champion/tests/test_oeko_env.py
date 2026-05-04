import sys
import os
import pytest
import numpy as np

# Modify sys.path to allow importing from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from oekolopoly.oekolopoly.envs.oeko_env import OekoEnv

def test_oeko_env_dtypes():
    env = OekoEnv()
    env.reset()
    assert env.init_v.dtype == np.int32
    assert env.V.dtype == np.int32
    assert env.Vmin.dtype == np.int32
    assert env.Vmax.dtype == np.int32
    assert env.Amin.dtype == np.int32
    assert env.Amax.dtype == np.int32
    assert env.curr_action.dtype == np.int32

def test_oeko_env_boundary_conditions():
    env = OekoEnv()

    # Test setting values at boundary points to see if clipping correctly enforces
    env.reset(options={"v": [29, 29, 29, 29, 29, 29, 48, 37, 30, 36]})
    assert np.array_equal(env.V, env.Vmax)

    # Test exactly 0 or 1 edge cases on some variables
    env.reset(options={"v": [1, 1, 1, 1, 1, 1, 1, -10, 0, 0]})
    assert np.array_equal(env.V, env.Vmin)

    # Let's mock env.V for clipping since reset() enforces observation bounds!
    env.V = np.array([0, 1, 1, 1, 1, 1, 1, -10, 0, 0], dtype=np.int32)
    assert env.clip(env.SANITATION) == 1

    env.V = np.array([35, 1, 1, 1, 1, 1, 1, -10, 0, 0], dtype=np.int32)
    assert env.clip(env.SANITATION) == 29

    # Test action point boundary (should be within 0-36)
    env.V = np.array([1, 1, 1, 1, 1, 1, 1, -10, 0, -5], dtype=np.int32)
    assert env.clip(env.POINTS) == 0

    env.V = np.array([1, 1, 1, 1, 1, 1, 1, -10, 0, 40], dtype=np.int32)
    assert env.clip(env.POINTS) == 36
