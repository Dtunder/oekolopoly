import os
import copy
import math
import numpy as np
import logging
import gymnasium as gym

logger = logging.getLogger("MCTSPlanner")

class MCTSNode:
    def __init__(self, obs, env_state_v, prior_prob=1.0, parent=None, action_from_parent=None):
        self.obs = obs
        self.env_state_v = np.copy(env_state_v)
        self.prior_prob = prior_prob
        self.parent = parent
        self.action_from_parent = action_from_parent

        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.done = False
        self.reward = 0.0

    @property
    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

class MCTSPlanner:
    def __init__(self, model, guardian, simulations=50, c_puct=1.0):
        self.model = model
        self.guardian = guardian
        self.simulations = simulations
        self.c_puct = c_puct

        # We need a clean env copy to simulate steps
        self.sim_env = gym.make("Oekolopoly-v2")

    def _clone_env_state(self, source_env):
        """Creates a shallow/deep copy of the environment state."""
        env_copy = copy.deepcopy(source_env)
        # Handle wrapped environments
        if hasattr(env_copy, 'env'):
            env_copy.env = copy.deepcopy(source_env.env)
        return env_copy

    def _evaluate_state(self, obs, lstm_states=None, episode_starts=None):
        """Use the PPO value network to evaluate a state."""
        import torch
        obs_tensor = self.model.policy.obs_to_tensor(np.array([obs]))[0]
        # In RecurrentPPO, predict_values needs proper hidden states and episode starts tensor
        if lstm_states is None:
            lstm_states = (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256))
        else:
            # model.predict returns numpy arrays for lstm_states, but predict_values needs torch tensors
            if isinstance(lstm_states[0], np.ndarray):
                lstm_states = (torch.tensor(lstm_states[0]).float(), torch.tensor(lstm_states[1]).float())
            else:
                lstm_states = (lstm_states[0].float(), lstm_states[1].float())

        if episode_starts is None:
            episode_starts_tensor = torch.tensor([1.0])
        else:
            # Need to handle bool arrays
            episode_starts_tensor = torch.tensor(np.array(episode_starts, dtype=np.float32)).float()

        with torch.no_grad():
            value = self.model.policy.predict_values(obs_tensor, lstm_states, episode_starts_tensor)[0].item()
        return value

    def search(self, root_obs, root_env):
        root = MCTSNode(root_obs, root_env.unwrapped.V)
        import torch
        root_lstm = (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256))
        root_episode_starts = np.ones((1,), dtype=bool)

        for _ in range(self.simulations):
            node = root

            # Use DeepCopyMCTSGymEnvWrapper technique
            import sys
            try:
                import oekolopoly.wrappers as wrappers
                DeepCopyWrapper = wrappers.DeepCopyMCTSGymEnvWrapper
            except (ImportError, AttributeError):
                class DeepCopyMCTSGymEnvWrapper(gym.Wrapper):
                    def __init__(self, env):
                        super().__init__(env)
                    def step(self, action):
                        return self.env.step(action)
                DeepCopyWrapper = DeepCopyMCTSGymEnvWrapper

            sim_env = self._clone_env_state(root_env)
            if not isinstance(sim_env, DeepCopyWrapper):
                # Ensure we are simulating on a wrapped version so unwrapped.done doesn't trigger immediately
                sim_env = DeepCopyWrapper(sim_env)
            lstm_states = root_lstm
            episode_starts = root_episode_starts

            # Selection
            while node.is_expanded and not node.done and len(node.children) > 0:
                best_ucb = -float('inf')
                best_child = None
                for child in node.children:
                    u = self.c_puct * child.prior_prob * math.sqrt(node.visits + 1e-8) / (1 + child.visits)
                    q = child.value
                    ucb = q + u
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = child

                node = best_child

                # Step the environment forward
                if node.action_from_parent is not None:
                    obs, reward, terminated, truncated, _ = sim_env.step(node.action_from_parent)
                    episode_starts = np.zeros((1,), dtype=bool)
                    node.done = terminated or truncated
                    node.reward = reward
                    node.obs = obs # update observation

            # Expansion
            if not node.done:
                # We need to generate legal actions. The SovereignGuardian is our guide.
                # First, get the guardian's primary suggested action.
                # For this, we first need a raw action from the PPO model.
                # PPO model requires batched shape for predict
                obs_batch = np.array([node.obs])
                raw_action, lstm_states = self.model.predict(
                    obs_batch, state=lstm_states, episode_start=episode_starts, deterministic=True
                )

                avail_ap = int(sim_env.unwrapped.V[9])

                # To create a search tree, we create multiple possible action distributions.
                # 1. The pure Guardian action
                # Note: SovereignGuardian in GUI has a different signature (avail) than in run_champion (raw_action, avail)
                if hasattr(self.guardian.get_final_action, '__code__') and self.guardian.get_final_action.__code__.co_argcount == 2:
                    g_res = self.guardian.get_final_action(avail_ap)
                else:
                    g_res = self.guardian.get_final_action(raw_action[0], avail_ap)

                if isinstance(g_res, tuple):
                    guardian_action = g_res[0]
                else:
                    guardian_action = g_res

                actions_to_expand = [(guardian_action, 0.8)] # High prior for the guardian

                # 2. Add some deterministic exploratory valid actions based on Guardian but tweaked
                # E.g. shift 1 point from QoL to Env if possible
                tweaked_action_1 = np.copy(guardian_action)
                if tweaked_action_1[3] > 0 and tweaked_action_1[5] < 5:
                    tweaked_action_1[3] -= 1
                    tweaked_action_1[5] += 1
                    actions_to_expand.append((tweaked_action_1, 0.1))

                tweaked_action_2 = np.copy(guardian_action)
                if tweaked_action_2[4] > 0 and tweaked_action_2[2] < 29:
                    tweaked_action_2[4] -= 1
                    tweaked_action_2[2] += 1
                    actions_to_expand.append((tweaked_action_2, 0.1))

                for act, prob in actions_to_expand:
                    child_node = MCTSNode(None, None, prior_prob=prob, parent=node, action_from_parent=act)
                    node.children.append(child_node)

                node.is_expanded = True

                # Evaluation (using the Value Network)
                # Note: We evaluate the current node, not the children (which haven't been stepped yet)
                # To do this correctly, we need the obs.
                value = self._evaluate_state(node.obs, lstm_states, episode_starts)
            else:
                # Terminal state
                year = sim_env.unwrapped.V[8]
                if year >= 30:
                    value = 100.0 # Success
                else:
                    value = -100.0 # Fatal failure

            # Backpropagation
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value_sum += value
                curr = curr.parent

        # Choose the best action
        if not root.children:
            raw_action, _ = self.model.predict(np.array([root_obs]), deterministic=True)
            avail_ap = int(root_env.unwrapped.V[9])
            if hasattr(self.guardian.get_final_action, '__code__') and self.guardian.get_final_action.__code__.co_argcount == 2:
                g_res = self.guardian.get_final_action(avail_ap)
            else:
                g_res = self.guardian.get_final_action(raw_action[0], avail_ap)

            if isinstance(g_res, tuple):
                final_act = g_res[0]
            else:
                final_act = g_res
            return final_act, []

        best_child = max(root.children, key=lambda c: c.visits)
        best_action = best_child.action_from_parent

        # Predict future trajectory (most visited path)
        trajectory = []
        curr = root
        temp_env = self._clone_env_state(root_env)
        for _ in range(5):
            if not curr.children:
                break
            best_curr_child = max(curr.children, key=lambda c: c.visits)
            if best_curr_child.action_from_parent is not None:
                _, _, done, _, _ = temp_env.step(best_curr_child.action_from_parent)
                trajectory.append(np.copy(temp_env.unwrapped.V))
                if done:
                    break
                # Mock a child node to continue trajectory if needed, though MCTS may not go this deep
                curr = best_curr_child
            else:
                break

        return best_action, trajectory
