import numpy as np
import gymnasium as gym
import copy
from typing import Dict, List, Tuple

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)
import torch.nn as nn
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init

from sb3_contrib import RecurrentPPO
from oekolopoly.wrappers import OekoActionBuilderWrapper, HomeostaticRewardV3
import oekolopoly.oekolopoly
from oekolopoly_gui import SovereignGuardian

class MCTSNode:
    def __init__(self, state, done=False, parent=None, action=None, prior_prob=1.0):
        self.state = state
        self.done = done
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}

    @property
    def q_value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def expand(self, action_probs: Dict[int, float], next_states_info: Dict[int, Tuple]):
        for action, prob in action_probs.items():
            if action not in self.children and prob > 0:
                next_state, done = next_states_info[action]
                self.children[action] = MCTSNode(
                    state=next_state,
                    done=done,
                    parent=self,
                    action=action,
                    prior_prob=prob
                )

def puct_score(parent: MCTSNode, child: MCTSNode, c_puct: float = 1.0) -> float:
    q_value = child.q_value
    puct = c_puct * child.prior_prob * np.sqrt(parent.visits) / (1 + child.visits)
    return q_value + puct

class PUCTPlanner:
    def __init__(self, env, model, num_simulations=50, c_puct=1.0):
        self.env = env
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.guardian = SovereignGuardian(self.env.unwrapped)

    def select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            best_action = None
            best_score = -float('inf')
            for action, child in node.children.items():
                score = puct_score(node, child, self.c_puct)
                if score > best_score:
                    best_score = score
                    best_action = action
            node = node.children[best_action]
        return node

    def get_action_mask(self, current_env):
        curr = current_env
        while hasattr(curr, 'env'):
            if hasattr(curr, 'valid_action_mask'):
                return curr.valid_action_mask()
            curr = curr.env
        if hasattr(curr, 'valid_action_mask'):
            return curr.valid_action_mask()
        return np.ones(9, dtype=bool)

    def evaluate_and_expand(self, node: MCTSNode, current_env) -> float:
        if node.done:
            year = current_env.unwrapped.V[8]
            return 1.0 if year >= 30 else -1.0

        # The model expects base env observations V[:10]
        obs = current_env.unwrapped.V[:10] - current_env.unwrapped.Vmin[:10]
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
            episode_starts = torch.ones((1,), dtype=torch.float32)

            features = self.model.policy.extract_features(obs_tensor)

            # value net evaluation
            num_layers = self.model.policy.lstm_actor.num_layers
            hidden_size_vf = self.model.policy.lstm_critic.hidden_size
            hx_vf = torch.zeros(num_layers, 1, hidden_size_vf)
            cx_vf = torch.zeros(num_layers, 1, hidden_size_vf)

            latent_vf, _ = self.model.policy._process_sequence(features, (hx_vf, cx_vf), episode_starts, self.model.policy.lstm_critic)
            latent_vf = self.model.policy.mlp_extractor.forward_critic(latent_vf)
            value = self.model.policy.value_net(latent_vf).item()

        action_mask = self.get_action_mask(current_env)
        num_valid = np.sum(action_mask)
        probs = action_mask.astype(float) / num_valid if num_valid > 0 else action_mask.astype(float)

        next_states_info = {}
        action_probs = {}
        for action in range(len(probs)):
            if probs[action] > 0:
                temp_env = copy.deepcopy(current_env)
                next_obs, reward, terminated, truncated, _ = temp_env.step(action)
                done = terminated or truncated

                # Predictive Action Pruning (Black Sky Shield)
                if action == 0:
                    raw_action = [
                        current_env.env._current_action_dict["Sanitation"],
                        current_env.env._current_action_dict["Production"],
                        current_env.env._current_action_dict["Education"],
                        current_env.env._current_action_dict["Quality of Life"],
                        current_env.env._current_action_dict["Population Growth"],
                        0
                    ]
                    trajectory = self.guardian.predict_future(raw_action, steps=5)
                    failed = False
                    for v in trajectory:
                        if v[3] > 35: # QoL explosion
                            failed = True
                            break
                    if failed:
                        probs[action] = 0.0
                        continue

                next_states_info[action] = (next_obs, done)
                action_probs[action] = probs[action]

        total_p = sum(action_probs.values())
        if total_p > 0:
            for a in action_probs:
                action_probs[a] /= total_p

        node.expand(action_probs, next_states_info)
        return value

    def backpropagate(self, node: MCTSNode, value: float):
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    def plan(self, initial_obs) -> int:
        root = MCTSNode(state=initial_obs)

        for _ in range(self.num_simulations):
            node = self.select(root)
            path = []
            curr = node
            while curr.parent is not None:
                path.append(curr.action)
                curr = curr.parent
            path.reverse()

            sim_env = copy.deepcopy(self.env)
            for a in path:
                sim_env.step(a)

            value = self.evaluate_and_expand(node, sim_env)
            self.backpropagate(node, value)

        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_action

def run_mcts_solver():
    env = gym.make("Oekolopoly-v2")
    env = OekoActionBuilderWrapper(env)
    env = HomeostaticRewardV3(env)

    obs, _ = env.reset()

    model = RecurrentPPO.load("Final_Sovereign_Champion/sota_recurrent_champion.zip")
    planner = PUCTPlanner(env, model, num_simulations=10) # 10 simulations for testing

    print("Starting MCTS planner...")
    done = False
    while not done:
        action = planner.plan(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("Final State at end of MCTS run:")
    print(env.unwrapped.V)
    print(f"Total Rounds: {env.unwrapped.V[8]}")

if __name__ == "__main__":
    run_mcts_solver()
