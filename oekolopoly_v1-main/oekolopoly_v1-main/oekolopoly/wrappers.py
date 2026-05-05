import math
import gym
import numpy as np


class OekoBoxActionWrapper(gym.ActionWrapper):
    def distribute1(self, action, points):
        action = list(action)
        action_sum = sum(action)

        if action_sum > 1:
            for i in range(len(action)):
                action[i] = action[i] / action_sum

        r = []
        for n in action:
            r.append(round(n * points))

        while sum(r) > points:
            max_index = r.index(max(r))
            r[max_index] -= 1

        assert sum(r) <= points

        return r


    def __init__(self, env):
        super().__init__(env)
        self.action_min = np.float32(np.array([0, -1,  0,  0,  0, -1]))
        self.action_max = np.float32(np.array([1,  1,  1,  1,  1,  1]))
        self.action_space = gym.spaces.Box(low=self.action_min, high=self.action_max)


    def action(self, act):
        act = np.clip(act, self.action_min, self.action_max, dtype=np.float32)
        assert self.action_space.contains(act), "Action not in action_space"

        if act[1] < 0:
            act[1] = -act[1]
            reduce_production = True
        else:
            reduce_production = False

        regions_act = act[0:5]
        special_act = round(act[5] * 5)
        regions_act = self.distribute1(regions_act, self.V[self.POINTS])
        if reduce_production: regions_act[1] = -regions_act[1]

        for i in range(len(regions_act)):
            region_result = self.V[i] + regions_act[i]
            if   region_result < self.Vmin[i]: regions_act[i] = self.Vmin[i] - self.V[i]
            elif region_result > self.Vmax[i]: regions_act[i] = self.Vmax[i] - self.V[i]

        act = np.append(regions_act, special_act)
        act -= self.Amin

        return act


class OekoSimpleActionWrapper(gym.ActionWrapper):
    ACTIONS = [
        '000000',
        '000010',
        '000020',
        '000030',
        '000100',
        '000110',
        '000120',
        '000200',
        '000210',
        '000300',
        '001000',
        '001010',
        '001020',
        '001100',
        '001110',
        '001200',
        '002000',
        '002010',
        '002100',
        '003000',
        '010000', '010001',
        '010010', '010011',
        '010020', '010021',
        '010100', '010101',
        '010110', '010111',
        '010200', '010201',
        '011000', '011001',
        '011010', '011011',
        '011100', '011101',
        '012000', '012001',
        '020000', '020001',
        '020010', '020011',
        '020100', '020101',
        '021000', '021001',
        '030000', '030001',
        '100000',
        '100010',
        '100020',
        '100100',
        '100110',
        '100200',
        '101000',
        '101010',
        '101100',
        '102000',
        '110000', '110001',
        '110010', '110011',
        '110100', '110101',
        '111000', '111001',
        '120000', '120001',
        '200000',
        '200010',
        '200100',
        '201000',
        '210000', '210001',
        '300000',
    ]

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.MultiDiscrete([77, 11])


    def action(self, act):
        action_index = act[0]
        extra_points = act[1]

        points = self.V[self.POINTS]
        act_string = self.ACTIONS[action_index]
        regions = [0, 0, 0, 0, 0]

        remaining = 0
        for i in range(5):
            region_points_float = points / 3 * int(act_string[i])
            region_points_int   = np.floor(region_points_float)
            remaining          += region_points_float - region_points_int
            regions[i]          = region_points_int
        remaining = round(remaining)

        if remaining:
            for i in range(5):
                if int(act_string[i]) > 0:
                    regions[i] += 1
                    remaining  -= 1
                    if remaining == 0: break
        assert remaining == 0

        if int(act_string[5]) == 1:
            regions[1] = -regions[1]

        for i in range(5):
            region_result = self.V[i] + regions[i]
            if   region_result < self.Vmin[i]: regions[i] = self.Vmin[i] - self.V[i]
            elif region_result > self.Vmax[i]: regions[i] = self.Vmax[i] - self.V[i]

        used_points = 0
        for i in range(5):
            used_points += abs(regions[i])

        assert used_points <= points

        act = np.append(regions, extra_points)
        for i in range(5): act[i] -= self.Amin[i]
        return act


class OekoSimpleObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.obs_count = 10  # how many of the original observations to use, starting from the first one
        self.obs_split = 3  # 3=low/mid/high

        self.original_observation_space = self.observation_space
        self.observation_space = gym.spaces.MultiDiscrete([3] * self.obs_count)


    def observation(self, obs):

        new_obs = [0] * self.obs_count
        for i in range(self.obs_count):
            new_obs[i] = math.floor(obs[i] / self.original_observation_space.nvec[i] * self.obs_split)

        return new_obs


class OekoBoxObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.low = np.array([
              1,  # 0 Sanitation
              1,  # 1 Production
              1,  # 2 Education
              1,  # 3 Quality of Life
              1,  # 4 Population Growth
              1,  # 5 Environment
              1,  # 6 Population
            -10,  # 7 Politics
              0,  # 8 Round
              0,  # 9 Actionpoints for next round
        ])
        self.high = np.array([
            29,  # 0 Sanitation
            29,  # 1 Production
            29,  # 2 Education
            29,  # 3 Quality of Life
            29,  # 4 Population Growth
            29,  # 5 Environment
            48,  # 6 Population
            37,  # 7 Politics
            30,  # 8 Round
            36,  # 9 Actionpoints for next round
        ])
        self.observation_space = gym.spaces.Box(self.low, self.high)


    def observation(self, obs):

        new_obs = obs + self.env.Vmin

        return new_obs



class OekoPerRoundRewardWrapper(gym.Wrapper):
    def __init__(self, env, per_round_reward=1):
        super().__init__(env)
        self.per_round_reward = per_round_reward

    def mod_reward(self):
        if self.done and self.V[self.ROUND] in range(10, 31):
            reward = self.balance
        else:
            reward = self.per_round_reward
        return reward

    def step(self, action):
        obs, _, done, d = self.env.step(action)
        reward = self.mod_reward()
        return obs, reward, done, d


class OekoAuxRewardWrapper(gym.Wrapper):
    def __init__(self, env, scaling=1):
        super().__init__(env)
        self.scaling = scaling

    def mod_reward(self):
        if self.done and self.V[self.ROUND] in range(10, 31):
            return self.balance
        else:
            production_reward = 14 - abs(15 - self.V[self.PRODUCTION])
            population_reward = 23 - abs(24 - self.V[self.POPULATION])
            return self.scaling * (production_reward + population_reward)

    def step(self, action):
        obs, _, done, d = self.env.step(action)
        reward = self.mod_reward()
        return obs, reward, done, d


class OekoBoxUnclippedActionWrapper(gym.ActionWrapper):
    def distribute1(self, action, points):
        action = list(action)

        r = []
        for n in action:
            r.append(round(n * points))

        return r


    def __init__(self, env):
        super().__init__(env)
        self.action_min = np.float32(np.array([0, -1,  0,  0,  0, -1]))
        self.action_max = np.float32(np.array([1,  1,  1,  1,  1,  1]))
        self.action_space = gym.spaces.Box(low=self.action_min, high=self.action_max)


    def action(self, act):
        act = np.clip(act, self.action_min, self.action_max, dtype=np.float32)
        assert self.action_space.contains(act), "Action not in action_space"

        if act[1] < 0:
            act[1] = -act[1]
            reduce_production = True
        else:
            reduce_production = False

        regions_act = act[0:5]
        special_act = round(act[5] * 5)
        regions_act = self.distribute1(regions_act, self.V[self.POINTS])
        if reduce_production: regions_act[1] = -regions_act[1]

        act = np.append(regions_act, special_act)
        act -= self.Amin
        act = np.clip(act, np.zeros(len(act)), self.Amax-self.Amin)
        return act
