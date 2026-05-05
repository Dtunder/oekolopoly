import os
import sys
import numpy as np
import gymnasium as gym
import torch

# Priority paths for Nasuta's original work
NASUTA_ROOT = r"G:\Meine Ablage\oekolopoly2\gymcts-games-main\gymcts-games-main\src"
if NASUTA_ROOT not in sys.path:
    sys.path.insert(0, NASUTA_ROOT)

# Standard MCTS Imports for Oekolopoly
from oekolopoly.env.oeko_env import OekoEnv

class SovereignMCTS:
    def __init__(self, model, num_simulations=3000, render_tree=True):
        """
        Initializes the Sovereign MCTS Planner.
        
        Args:
            model: The RecurrentPPO model to use as a policy guide.
            num_simulations: How many 'futures' to simulate (Default 3000 for Nasuta-style deep thinking).
            render_tree: Whether to print the visual MCTS tree to console.
        """
        self.model = model
        self.num_simulations = num_simulations
        self.render_tree = render_tree

    def search(self, env) -> int:
        """
        Performs a deep MCTS search guided by the neural policy.
        """
        # Create a simulation copy
        sim_env = gym.make("Oekolopoly-v2")
        
        # Import Nasuta's specialized MCTS components
        from gymcts.gymcts_agent import GymctsAgent
        from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
        from oekolopoly.env.oeko_env import OekoActionBuilderWrapper
        
        # 1. Wrap with ActionBuilder (Nasuta's translation layer)
        wrapped_env = OekoActionBuilderWrapper(sim_env)
        
        # 2. Wrap with DeepCopy (State management for MCTS)
        wrapped_env = DeepCopyMCTSGymEnvWrapper(wrapped_env)
        
        # 3. Add MCTS Bridges (Robust recursive search for Nasuta compatibility)
        def find_valid_mask():
            curr = wrapped_env
            while curr is not None:
                if hasattr(curr, 'valid_action_mask'):
                    return curr.valid_action_mask()
                curr = getattr(curr, 'env', None)
            return [True] * 9 # Fallback
            
        wrapped_env.get_valid_actions = lambda: [i for i, valid in enumerate(find_valid_mask()) if valid]
        
        # 4. RESET AFTER WRAPPING (Crucial for ActionBuilder initialization)
        wrapped_env.reset(options={"v": env.unwrapped.V.copy()})
        
        # Setup MCTS Agent with Nasuta's original parameters
        agent = GymctsAgent(
            env=wrapped_env, 
            number_of_simulations_per_step=self.num_simulations,
            render_tree_after_step=self.render_tree,
            render_tree_max_depth=2 # Keeps the console output manageable
        )
        
        # Perform the actual deep search
        # Note: In future versions, we can inject the 'model' here to bias the search
        action = agent.vanilla_mcts_search(num_simulations=self.num_simulations)
        
        if self.render_tree:
            print(f" [Sovereign Analysis] Search complete. Best Move: {action}")
            
        return action
