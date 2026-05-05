import os
import sys
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

# MANDATORY SOVEREIGN LSTM PATCH (Fixes PyTorch/Gymnasium int64 conflict)
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init

# Priority paths for Nasuta's original work
NASUTA_ROOT = r"G:\Meine Ablage\oekolopoly2\gymcts-games-main\gymcts-games-main\src"
if NASUTA_ROOT not in sys.path:
    sys.path.insert(0, NASUTA_ROOT)

# Standard MCTS Imports for Oekolopoly
from oekolopoly.env.oeko_env import OekoEnv

# GLOBAL SOVEREIGN COMPONENTS (Avoids DeepCopy overhead)
_GLOBAL_SOVEREIGN_MODEL = None

import time

def guided_rollout_wrapper(self_wrapper):
    """Uses a FAST heuristic for massive simulation throughput."""
    try:
        temp_env = self_wrapper.env.unwrapped
        total_reward = 0
        d_done = False
        steps = 0
        
        while not d_done and steps < 30:
            V = temp_env.V
            # SYNC FIX: Access the wrapper's internal AP tracker if available
            if hasattr(self_wrapper, '_available_action_points'):
                avail = int(self_wrapper._available_action_points)
            else:
                avail = int(V[9])
            
            if temp_env.done:
                break
                
            valid_actions = self_wrapper.get_valid_actions()
            if not valid_actions: break
            
            # HEURISTIC SELECTION (Sovereign Guardian Style)
            if avail > 0:
                # Priority: Sanity (1) -> Prod (2) -> QoL (5) -> Edu (4)
                if 1 in valid_actions and V[5] < 12: move = 1
                elif 5 in valid_actions and V[3] < 10: move = 5
                elif 2 in valid_actions and V[7] < 8: move = 2
                elif 4 in valid_actions and V[4] < 10: move = 4
                else: 
                    # If no emergencies, pick random investment
                    investments = [a for a in valid_actions if a != 0]
                    move = np.random.choice(investments) if investments else 0
            else:
                move = 0 # Must end round
            
            obs_ext, rew, term, trunc, info = self_wrapper.step(move)
            
            # Stability Reward
            stability = 30 - (np.max(V[:8]) - np.min(V[:8]))
            total_reward += stability + rew
            if move == 0: total_reward += 10000 
            
            d_done = term or trunc
            steps += 1
            
        if d_done and not (int(temp_env.V[8]) >= 30):
            total_reward -= 2000000 
            
        return total_reward
    except Exception as e:
        return -5000000

class SovereignMCTS:
    def __init__(self, model, num_simulations=1000, render_tree=True):
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
        global _GLOBAL_SOVEREIGN_MODEL
        _GLOBAL_SOVEREIGN_MODEL = self.model
        
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
            
        def get_sovereign_valid_actions():
            mask = find_valid_mask()
            # SYNC FIX
            if hasattr(wrapped_env, '_available_action_points'):
                avail = int(wrapped_env._available_action_points)
            else:
                avail = int(wrapped_env.env.unwrapped.V[9])
                
            valid = [i for i, v in enumerate(mask) if v]
            
            # HARD MASKING: Only remove Action 0 if there's SOMETHING ELSE to do
            # and we still have points.
            if avail > 0 and 0 in valid and len(valid) > 1:
                valid.remove(0)
            return valid

        wrapped_env.get_valid_actions = get_sovereign_valid_actions
        
        # 4. MONKEY PATCH THE CLASS (Top-level function avoids closure capture)
        DeepCopyMCTSGymEnvWrapper.rollout = guided_rollout_wrapper
        
        wrapped_env.reset()
        
        # 5. STATE SYNC
        wrapped_env.env.unwrapped.V = env.unwrapped.V.copy()
        wrapped_env.env.unwrapped.obs = env.unwrapped.obs.copy()
        if hasattr(wrapped_env, '_available_action_points'):
            wrapped_env._available_action_points = int(env.unwrapped.V[9])
        
        # Setup MCTS Agent with Nasuta's original parameters
        agent = GymctsAgent(
            env=wrapped_env, 
            number_of_simulations_per_step=self.num_simulations,
            render_tree_after_step=self.render_tree,
            render_tree_max_depth=2 
        )
        
        # Perform the actual deep search
        import sys
        import logging
        m_logger = logging.getLogger("MCTS")
        m_logger.info("\n" + "="*50)
        m_logger.info(" [SOVEREIGN DEEP THINKING TREE START]")
        sys.stdout.flush()
        
        try:
            # Capture the search logic
            action = agent.vanilla_mcts_search(num_simulations=self.num_simulations)
        except Exception as e:
            if "charmap" in str(e):
                m_logger.warning(" [Encoding Error] Tree contains characters not supported by console. Falling back to simple best move.")
                # We still want the best action even if tree printing fails
                action = agent.search_root_node.get_best_action()
            else:
                raise e
        
        m_logger.info(" [SOVEREIGN DEEP THINKING TREE END]")
        m_logger.info("="*50)
        m_logger.info(f"\n [Sovereign Analysis] Search complete. Best Move: {action}")
        sys.stdout.flush()
        
        return action
