import numpy as np
import gymnasium as gym
try:
    from gymcts.gymcts_agent import GymctsAgent
    from gymcts.gymcts_action_history_wrapper import ActionHistoryMCTSGymEnvWrapper
except ImportError as e:
    print(f"Error importing gymcts: {e}")
    print("Ensure gymcts is installed. Trying to continue...")

from oekolopoly.wrappers import OekoActionBuilderWrapper, HomeostaticRewardV3

# Registration of the environment if not already done
# In oekolopoly_v2, it seems to be registered in oekolopoly/__init__.py
import oekolopoly.oekolopoly

def run_mcts_solver():
    # 1. Initialize environment with sequential action builder and homeostatic reward
    env = gym.make("Oekolopoly-v2")
    env = OekoActionBuilderWrapper(env)
    env = HomeostaticRewardV3(env)
    
    def action_mask_fn(env):
        # Traverse wrappers to find the one with valid_action_mask
        curr = env
        while hasattr(curr, 'env'):
            if hasattr(curr, 'valid_action_mask'):
                return curr.valid_action_mask()
            curr = curr.env
        if hasattr(curr, 'valid_action_mask'):
            return curr.valid_action_mask()
        return None

    # 2. Wrap for MCTS
    env = ActionHistoryMCTSGymEnvWrapper(env, action_mask_fn=action_mask_fn)
    env.reset()

    # 3. Create the Gymcts Agent
    agent = GymctsAgent(
        env=env,
        clear_mcts_tree_after_step=True,
        render_tree_after_step=False,
        number_of_simulations_per_step=1000, # Reduced for testing
    )

    # 4. Solve
    print("Starting MCTS solver...")
    agent.solve()
    
    print("Final State at end of MCTS run:")
    print(env.unwrapped.V)
    print(f"Total Rounds: {env.unwrapped.V[8]}")

if __name__ == "__main__":
    run_mcts_solver()
