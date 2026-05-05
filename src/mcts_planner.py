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
        
        # 4. SOVEREIGN GUIDED ROLLOUTS (Neural Guidance)
        def guided_rollout():
            """Uses the RecurrentPPO model to play a meaningful future."""
            # We don't call reset here because GymctsAgent handles the state
            temp_obs = wrapped_env.env.unwrapped.obs.copy() # Get raw obs
            
            # Action Mapping (Nasuta-Discrete -> Vector Index)
            # 1: San+, 2: Prod+, 4: Edu+, 5: QoL+, 6: PG+
            mapping = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}
            
            total_reward = 0
            d_done = False
            steps = 0
            l_states = None
            e_starts = np.ones((1,), dtype=bool)
            
            while not d_done and steps < 25:
                # 1. Get Model Intuition (The 'Vector' action)
                # CLIP OBSERVATION FOR MODEL (Model only knows Rounds 0-30)
                model_obs = temp_obs.copy()
                if len(model_obs) >= 9:
                    model_obs[8] = min(model_obs[8], 30)
                
                act_vector, l_states = self.model.predict(model_obs, state=l_states, episode_start=e_starts, deterministic=False)
                
                # 2. Translate Vector to the best Discrete Action
                valid_actions = wrapped_env.get_valid_actions()
                if not valid_actions: break
                
                # Biasing logic: If model wants to spend points in a category, prioritize that discrete action
                weights = np.ones(len(valid_actions))
                for i, v_act in enumerate(valid_actions):
                    if v_act in mapping:
                        target_idx = mapping[v_act]
                        if act_vector[target_idx] > 0:
                            weights[i] = 10.0 # Heavy bias towards model's choice
                    elif v_act == 0: # Next Round
                        if np.sum(act_vector[:5]) == 0:
                            weights[i] = 5.0 # Bias towards finishing round if no points spent
                
                # Sample based on weights
                weights /= np.sum(weights)
                move = np.random.choice(valid_actions, p=weights)
                
                # 3. Step
                obs_ext, rew, term, trunc, _ = wrapped_env.step(move)
                temp_obs = wrapped_env.env.unwrapped.obs.copy() # Sync for next iteration
                
                total_reward += rew
                d_done = term or trunc
                steps += 1
                e_starts = np.zeros((1,), dtype=bool)
                
            return total_reward

        wrapped_env.rollout = guided_rollout
        
        # 5. RESET AFTER WRAPPING (Crucial for ActionBuilder initialization)
        wrapped_env.reset(options={"v": env.unwrapped.V.copy()})
        
        # Setup MCTS Agent with Nasuta's original parameters
        agent = GymctsAgent(
            env=wrapped_env, 
            number_of_simulations_per_step=self.num_simulations,
            render_tree_after_step=self.render_tree,
            render_tree_max_depth=2 # Keeps the console output manageable
        )
        
        # Perform the actual deep search
        import sys
        import logging
        m_logger = logging.getLogger("MCTS")
        m_logger.info("\n" + "="*50)
        m_logger.info(" [SOVEREIGN DEEP THINKING TREE START]")
        sys.stdout.flush()
        
        action = agent.vanilla_mcts_search(num_simulations=self.num_simulations)
        
        sys.stdout.flush()
        m_logger.info(" [SOVEREIGN DEEP THINKING TREE END]")
        m_logger.info("="*50 + "\n")
        
        if self.render_tree:
            m_logger.info(f" [Sovereign Analysis] Search complete. Best Move: {action}")
            
        return action
