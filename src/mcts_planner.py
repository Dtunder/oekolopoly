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
    """Uses the GLOBAL model reference with timing."""
    start_sim = time.time()
    try:
        global _GLOBAL_SOVEREIGN_MODEL
        if _GLOBAL_SOVEREIGN_MODEL is None:
            return -1000000
            
        temp_obs = self_wrapper.env.unwrapped.obs.copy()
        mapping = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}
        
        total_reward = 0
        d_done = False
        steps = 0
        l_states = None
        e_starts = np.ones((1,), dtype=bool)
        
        while not d_done and steps < 15:
            V = self_wrapper.env.unwrapped.V
            avail = int(V[9])
            
            if self_wrapper.env.unwrapped.done:
                break
            
            model_obs = temp_obs.copy()
            if len(model_obs) >= 9:
                model_obs[8] = min(model_obs[8], 30)
            
            # USE GLOBAL MODEL
            act_vector, l_states = _GLOBAL_SOVEREIGN_MODEL.predict(model_obs, state=l_states, episode_start=e_starts, deterministic=False)
            
            valid_actions = self_wrapper.get_valid_actions()
            if not valid_actions: break
            
            weights = np.ones(len(valid_actions))
            for i, v_act in enumerate(valid_actions):
                if v_act == 0:
                    weights[i] = 0.0 if avail > 0 else 100.0
                elif v_act in mapping:
                    target_idx = mapping[v_act]
                    if act_vector[target_idx] > 0: weights[i] = 20.0 
                    if v_act == 5 and V[3] < 10: weights[i] += 500.0
                    if v_act == 1 and V[5] < 12: weights[i] += 500.0
                    if v_act == 2 and V[7] < 8:  weights[i] += 500.0
            
            if np.sum(weights) == 0: weights = np.ones(len(valid_actions))
            weights /= np.sum(weights)
            move = np.random.choice(valid_actions, p=weights)
            
            obs_ext, rew, term, trunc, info = self_wrapper.step(move)
            temp_obs = self_wrapper.env.unwrapped.obs.copy()
            
            stability = 30 - (np.max(V[:8]) - np.min(V[:8]))
            total_reward += stability + rew
            if move == 0: total_reward += 10000 
            
            d_done = term or trunc
            steps += 1
            e_starts = np.zeros((1,), dtype=bool)
            
        if d_done and not (int(self_wrapper.env.unwrapped.V[8]) >= 30):
            total_reward -= 2000000 
            
        print(f"Sim done in {time.time() - start_sim:.3f}s")
        return total_reward
    except Exception as e:
        return -5000000

class SovereignMCTS:
    def __init__(self, model, num_simulations=50, render_tree=True):
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
        
        action = agent.vanilla_mcts_search(num_simulations=self.num_simulations)
        
        m_logger.info(" [SOVEREIGN DEEP THINKING TREE END]")
        m_logger.info("="*50)
        m_logger.info(f"\n [Sovereign Analysis] Search complete. Best Move: {action}")
        sys.stdout.flush()
        
        return action
