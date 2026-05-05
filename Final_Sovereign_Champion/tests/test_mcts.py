import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from mcts_planner import run_mcts_solver

def test_mcts_runs():
    try:
        run_mcts_solver()
    except Exception as e:
        assert False, f"MCTS planner failed with exception: {e}"

if __name__ == '__main__':
    test_mcts_runs()
