import unittest
import numpy as np

import sys
sys.path.insert(0, 'src')
from oekolopoly.oekolopoly.envs.oeko_env import OekoEnv

class TestEquivalence(unittest.TestCase):
    def setUp(self):
        self.env = OekoEnv()

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(len(obs), 10)
        self.assertEqual(info['balance'], 0)

    def test_step_valid_moves(self):
        self.env.reset(seed=42)

        # Test a sequence of valid moves - actions are in shifted space [0..max]
        action = np.array([0, 28+0, 0, 0, 0, 5])

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertFalse(terminated)
        self.assertEqual(info['valid_move'], True)

    def test_step_invalid_action_points(self):
        self.env.reset()

        # Too many action points
        action = np.array([20, 20+28, 20, 20, 20, 5])
        obs, reward, terminated, truncated, info = self.env.step(action.copy())

        self.assertEqual(terminated, True)
        self.assertEqual(info['valid_move'], False)

if __name__ == '__main__':
    unittest.main()
