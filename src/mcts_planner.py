import os
import sys
import copy
import math
import numpy as np
import gymnasium as gym
import torch
from sb3_contrib import RecurrentPPO

# Ensure paths
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import oekolopoly.oekolopoly
from run_champion import SovereignGuardian

# MCTS Node
class PUCTNode:
    def __init__(self, state, obs, prior_prob, parent=None, action=None):
        self.state = state  # Tuple of (V, curr_action) to restore env
        self.obs = obs      # Raw observation for PPO
        self.prior_prob = prior_prob
        self.parent = parent
        self.action = action  # Action taken to reach this node (in base action space)
        self.children = {}    # action -> PUCTNode
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

def save_env_state(env):
    """Saves the minimal required state of OekoEnv."""
    return (env.unwrapped.V.copy(), env.unwrapped.curr_action.copy())

def load_env_state(env, state):
    """Loads the minimal required state into OekoEnv."""
    env.unwrapped.V = state[0].copy()
    env.unwrapped.curr_action = state[1].copy()
    env.unwrapped.obs = env.unwrapped.V - env.unwrapped.Vmin
    return env.unwrapped.obs

class SovereignMCTS:
    def __init__(self, model, c_puct=1.5, num_simulations=100, num_samples=5):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.num_samples = num_samples # Max branches per node

    def search(self, env):
        # Initial node
        root_state = save_env_state(env)
        root_obs = env.unwrapped.obs.copy()
        root = PUCTNode(root_state, root_obs, prior_prob=1.0)

        # Temp env for simulations
        sim_env = gym.make("Oekolopoly-v2")
        sim_env.reset() # Reset once to initialize
        guardian = SovereignGuardian(sim_env)

        for _ in range(self.num_simulations):
            node = root
            load_env_state(sim_env, node.state)

            # Selection
            while node.is_expanded and len(node.children) > 0:
                best_score = -float('inf')
                best_action = None
                best_child = None

                for action, child in node.children.items():
                    # PUCT Formula
                    q_val = child.value()
                    u_val = self.c_puct * child.prior_prob * math.sqrt(node.visits) / (1 + child.visits)
                    score = q_val + u_val

                    if score > best_score:
                        best_score = score
                        best_action = action
                        best_child = child

                if best_child is None:
                    break

                node = best_child
                load_env_state(sim_env, node.state)

            # Check if game over
            v_array = sim_env.unwrapped.V
            terminated = v_array[8] >= 30 or v_array[8] < sim_env.unwrapped.Vmin[8] # Game limits

            # Since Oekolopoly-v2 returns done in step, let's also check points
            # If terminated, just backup
            if not terminated and not node.is_expanded:
                # Expand
                obs_tensor = torch.tensor(node.obs).unsqueeze(0)
                with torch.no_grad():
                    # Setup dummy states for recurrent policy
                    # RecurrentPPO expects num_layers = 1 (usually) but sometimes 2 depending on config.
                    # We can fetch num_layers from the lstm.
                    n_layers_actor = self.model.policy.lstm_actor.num_layers
                    n_layers_critic = self.model.policy.lstm_critic.num_layers
                    lstm_states_actor = tuple(torch.zeros(n_layers_actor, 1, self.model.policy.lstm_actor.hidden_size) for _ in range(2))
                    lstm_states_critic = tuple(torch.zeros(n_layers_critic, 1, self.model.policy.lstm_critic.hidden_size) for _ in range(2))
                    episode_starts = torch.tensor([False], dtype=torch.bool, device=obs_tensor.device)

                    # Evaluate Value
                    value_pred = self.model.policy.predict_values(obs_tensor, lstm_states_critic, episode_starts)
                    value = value_pred.item()

                    # Policy Distribution
                    dist_tuple = self.model.policy.get_distribution(obs_tensor, lstm_states_actor, episode_starts)
                    dist = dist_tuple[0] # The distribution is the first element

                # Sample candidate actions
                candidates = []
                for _ in range(self.num_samples * 2): # Oversample to find unique ones
                    # Handle continuous actions, we should probably round them if discrete or use them directly
                    # stable_baselines distributions sample() directly
                    act = dist.sample()
                    # convert back to array
                    act_arr = act.cpu().numpy()[0]
                    # Convert to tuple to make it hashable
                    act_tup = tuple(act_arr)
                    if act_tup not in [c[0] for c in candidates]:
                        log_prob = dist.log_prob(act).sum().item() # log_prob might be a tensor of shape [1, action_dim], so sum it
                        candidates.append((act_tup, act_arr, math.exp(log_prob)))
                        if len(candidates) >= self.num_samples:
                            break

                # Predictive Action Pruning using SovereignGuardian
                node.is_expanded = True

                for act_tup, act_arr, prob in candidates:
                    # Rollout 5 years to check for collapse
                    # Reset sim_env to node state
                    load_env_state(sim_env, node.state)

                    # The PPO model returns a raw continuous action which Guardian converts
                    # We must pass the raw action to guardian to get the true valid action
                    avail = int(sim_env.unwrapped.V[9])
                    final_action = guardian.get_final_action(act_arr, avail)

                    # Apply action
                    obs, r, term, trunc, info = sim_env.step(final_action)

                    collapse = False
                    if term or trunc:
                        # Ensure we don't prune an immediate victory
                        if sim_env.unwrapped.V[8] < 30:
                            collapse = True
                    else:
                        # Forward simulate up to 4 more years using Guardian
                        for step in range(4):
                            avail = int(sim_env.unwrapped.V[9])
                            # Dummy action since Guardian overwrites anyway
                            dummy = np.zeros(6, dtype=int)
                            g_action = guardian.get_final_action(dummy, avail)
                            obs, r, term, trunc, info = sim_env.step(g_action)
                            if term or trunc:
                                # Did it collapse or reach victory?
                                if sim_env.unwrapped.V[8] < 30:
                                    collapse = True
                                break

                    if collapse:
                        prob = 0.0 # PRUNE!

                    if prob > 0:
                        # Re-run the single action to get the child state
                        load_env_state(sim_env, node.state)
                        # We use final_action here as well
                        avail = int(sim_env.unwrapped.V[9])
                        final_action = guardian.get_final_action(act_arr, avail)
                        child_obs, _, _, _, _ = sim_env.step(final_action)
                        child_state = save_env_state(sim_env)
                        # the key is the raw action tuple (what PPO predicted) so selection works
                        node.children[act_tup] = PUCTNode(child_state, child_obs.copy(), prob, parent=node, action=act_tup)

            else:
                # If terminal, value is based on success
                if v_array[8] >= 30:
                    value = 1.0
                else:
                    value = -1.0

            # Backpropagate
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value_sum += value
                curr = curr.parent

        # Select best action based on visit counts
        best_action = None
        best_visits = -1
        for act, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = act

        # If all branches pruned or zero children, fallback to guardian directly
        if best_action is None:
            print("All MCTS branches pruned! Falling back to Guardian directly.")
            avail = int(env.unwrapped.V[9])
            guardian = SovereignGuardian(env)
            best_action = tuple(guardian.get_final_action(np.zeros(6, dtype=int), avail))

        return np.array(best_action)


def run_mcts_planner():
    # 1. Initialize environment
    env = gym.make("Oekolopoly-v2")

    # 2. Load Model
    model_path = os.path.join(ROOT, "sota_recurrent_champion.zip")
    model = RecurrentPPO.load(model_path, device='cpu')

    # 3. Create MCTS Planner
    planner = SovereignMCTS(model, c_puct=1.5, num_simulations=50, num_samples=5)

    obs, _ = env.reset()

    print("Starting Sovereign MCTS execution...")

    for year in range(35):
        # Get action from MCTS
        action = planner.search(env)

        # The planner returns raw action, must map with guardian
        avail = int(env.unwrapped.V[9])
        guardian = SovereignGuardian(env)
        final_action = guardian.get_final_action(action, avail)

        # Step env
        obs, reward, terminated, truncated, info = env.step(final_action)
        V = env.unwrapped.V

        print(f"Year {int(V[8])}: Env={int(V[5])}, QoL={int(V[3])}, AP={int(V[9])}")

        if terminated or truncated:
            break

    print("Final State at end of MCTS run:")
    print(env.unwrapped.V)
    print(f"Total Rounds: {env.unwrapped.V[8]}")

if __name__ == "__main__":
    # Same patch as run_champion to avoid LSTM init bug
    original_lstm_init = torch.nn.LSTM.__init__
    def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
        return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
    torch.nn.LSTM.__init__ = patched_lstm_init

    run_mcts_planner()
