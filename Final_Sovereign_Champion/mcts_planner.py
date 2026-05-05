import numpy as np
import gymnasium as gym
from sb3_contrib import RecurrentPPO
import torch

# WATERPROOF MONKEY PATCH for LSTM
import torch.nn as nn
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init

import copy
from oekolopoly.wrappers import OekoActionBuilderWrapper, HomeostaticRewardV3
import oekolopoly.oekolopoly
from oekolopoly_gui import SovereignGuardian

class MCTSNode:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.reward = 0
        self.is_expanded = False

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTSPlanner:
    def __init__(self, env, model, num_simulations=50, c_puct=1.0):
        self.env = env
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.guardian = SovereignGuardian(self.env)

    def get_action_mask(self, env):
        curr = env
        while hasattr(curr, 'env'):
            if hasattr(curr, 'valid_action_mask'):
                return curr.valid_action_mask()
            curr = curr.env
        if hasattr(curr, 'valid_action_mask'):
            return curr.valid_action_mask()
        return None

    def plan(self, observation):
        root = MCTSNode(prior=0, to_play=-1)
        root.state = observation

        for _ in range(self.num_simulations):
            node = root
            sim_env = copy.deepcopy(self.env)
            search_path = [node]

            # Selection
            while node.expanded():
                action, node = self.select_child(node)
                obs, reward, terminated, truncated, _ = sim_env.step(action)
                search_path.append(node)
                if terminated or truncated:
                    break

            # Expansion and Evaluation
            obs = root.state if not node.expanded() else obs
            value = self.evaluate_and_expand(node, sim_env, obs)

            # Backpropagation
            self.backpropagate(search_path, value, sim_env.unwrapped.to_play if hasattr(sim_env.unwrapped, 'to_play') else -1)

        return self.select_action(root)

    def select_child(self, node):
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = self.ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, parent, child):
        prior_score = self.c_puct * child.prior * np.sqrt(parent.visit_count) / (child.visit_count + 1)
        if child.visit_count > 0:
            value_score = child.value()
        else:
            value_score = 0
        return value_score + prior_score

    def evaluate_and_expand(self, node, sim_env, obs):
        mask = self.get_action_mask(sim_env)
        if mask is None:
            mask = np.ones(sim_env.action_space.n)

        with torch.no_grad():
            # Get V (first 10 elements of the state match the unwrapped env state size typically,
            # but model was trained on unwrapped obs).
            # The wrapped observation has size 16 (state + action history),
            # while base env has size 10 (just V).
            v_state = sim_env.unwrapped.V - sim_env.unwrapped.Vmin

            # Prepare tensor for PPO
            # Normalization might be required depending on how PPO was trained, assuming direct V
            obs_tensor = torch.tensor(v_state).float().unsqueeze(0).to(self.model.device)

            # Extract features manually to avoid shape mismatches with get_distribution
            features = self.model.policy.extract_features(obs_tensor)

            # Since RecurrentPPO, features go through LSTM. We use zero states for simulation.
            lstm_states = (
                torch.zeros(self.model.policy.lstm_actor.num_layers, 1, self.model.policy.lstm_actor.hidden_size).to(self.model.device),
                torch.zeros(self.model.policy.lstm_actor.num_layers, 1, self.model.policy.lstm_actor.hidden_size).to(self.model.device),
            )
            episode_starts = torch.tensor([1.0]).to(self.model.device)

            latent_pi, _ = self.model.policy._process_sequence(features, lstm_states, episode_starts, self.model.policy.lstm_actor)

            # Pass through action net to get action distribution parameters (mean for Box)
            action_mean = self.model.policy.action_net(latent_pi)

            # Also get value
            latent_vf, _ = self.model.policy._process_sequence(features, lstm_states, episode_starts, self.model.policy.lstm_critic)
            value = self.model.policy.value_net(latent_vf).item()

            # Normalize to probabilities using softmax to get priors for the 6 sequential actions
            action_logits = action_mean.squeeze(0).cpu().numpy()
            # action_logits has shape (6,), mask has shape (9,) for sequential builder.
            # We map 6 base dimensions to the 9 builder branches.
            # 0: Next Round, 1: San, 2: Prod+, 3: Prod-, 4: Edu, 5: QoL, 6: Pop Growth, 7: Pop Extra+, 8: Pop Extra-
            # Let us craft a compatible 9-dim logits vector based on PPO's 6-dim Box desires.
            # Base action is Box(0,1) interpreted roughly as proportion.
            mapped_logits = np.zeros(9)
            mapped_logits[0] = 0.1 # Base bias to finish round if no points
            mapped_logits[1] = action_logits[0]
            mapped_logits[2] = action_logits[1] if action_logits[1] > 0 else 0
            mapped_logits[3] = -action_logits[1] if action_logits[1] < 0 else 0
            mapped_logits[4] = action_logits[2]
            mapped_logits[5] = action_logits[3]
            mapped_logits[6] = action_logits[4]
            mapped_logits[7] = action_logits[5] if action_logits[5] > 0 else 0
            mapped_logits[8] = -action_logits[5] if action_logits[5] < 0 else 0
            action_logits = mapped_logits
            # Action space is Box(0, 1, (6,)). The PPO gives logits/mean. We softmax it for MCTS discrete priors.
            exp_logits = np.exp(action_logits - np.max(action_logits))
            probs = exp_logits / exp_logits.sum()

        probs = probs * mask
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = mask / mask.sum()

        # PREDICTIVE ACTION PRUNING (Black Sky Shield)
        for action in range(sim_env.action_space.n):
            if mask[action]:
                prune_env = copy.deepcopy(sim_env)
                _, _, term, trunc, info = prune_env.step(action)

                prune_prob = probs[action]
                if term and prune_env.unwrapped.V[8] < 30:
                    prune_prob = 0.0 # Strict prune
                elif not term and not trunc:
                    if hasattr(prune_env, 'points_to_distribute') and prune_env.points_to_distribute == 0:
                        guardian = SovereignGuardian(prune_env)
                        current_action = prune_env.unwrapped.prev_action - prune_env.unwrapped.Amin
                        try:
                            trajectory = guardian.predict_future(current_action, steps=5)
                            for step_v in trajectory:
                                if step_v[8] < 30:
                                    if step_v[3] <= 0 or step_v[5] <= 0 or step_v[6] >= 40:
                                        prune_prob = 0.0
                                        break
                        except Exception:
                            pass

                if prune_prob > 0:
                    node.children[action] = MCTSNode(prior=prune_prob, to_play=sim_env.unwrapped.to_play if hasattr(sim_env.unwrapped, 'to_play') else -1)

        node.is_expanded = True
        return value

    def backpropagate(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

    def select_action(self, root):
        if not root.children:
            mask = self.get_action_mask(self.env)
            valid_actions = np.where(mask == 1)[0]
            return np.random.choice(valid_actions)
        visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
        visit_counts.sort(reverse=True)
        return visit_counts[0][1]

def run_mcts_solver():
    env = gym.make("Oekolopoly-v2")
    env = OekoActionBuilderWrapper(env)
    env = HomeostaticRewardV3(env)
    obs, _ = env.reset()

    # Load model with base env to match its training space
    model = RecurrentPPO.load("sota_recurrent_champion.zip", device="cpu")

    planner = MCTSPlanner(env, model, num_simulations=50)

    print("Starting MCTS solver with PPO Priors & Black Sky Shield...")
    done = False
    while not done:
        action = planner.plan(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        planner.env = copy.deepcopy(env) # Update planner env
        done = terminated or truncated

    print("Final State at end of MCTS run:")
    print(env.unwrapped.V)
    print(f"Total Rounds: {env.unwrapped.V[8]}")

if __name__ == "__main__":
    run_mcts_solver()
