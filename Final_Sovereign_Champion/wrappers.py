import math
import gymnasium as gym
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
        regions_act = self.get_wrapper_attr('distribute1')(regions_act, self.unwrapped.V[self.unwrapped.POINTS])
        if reduce_production:
            regions_act[1] = -regions_act[1]

        for i in range(len(regions_act)):
            region_result = self.unwrapped.V[i] + regions_act[i]
            if   region_result < self.unwrapped.Vmin[i]: regions_act[i] = self.unwrapped.Vmin[i] - self.unwrapped.V[i]
            elif region_result > self.unwrapped.Vmax[i]: regions_act[i] = self.unwrapped.Vmax[i] - self.unwrapped.V[i]

        act = np.append(regions_act, special_act)
        act -= self.unwrapped.Amin

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

        points = self.unwrapped.V[self.unwrapped.POINTS]
        act_string = self.ACTIONS[action_index]
        regions = [0, 0, 0, 0, 0]

        remaining = 0
        for i in range(5):
            region_points_float = points / 3 * int(act_string[i])
            region_points_int   = np.int64(np.floor(region_points_float))  # /WK/2025-11/ bug fix: np.int64
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
            region_result = self.unwrapped.V[i] + regions[i]
            if   region_result < self.unwrapped.Vmin[i]: regions[i] = self.unwrapped.Vmin[i] - self.unwrapped.V[i]
            elif region_result > self.unwrapped.Vmax[i]: regions[i] = self.unwrapped.Vmax[i] - self.unwrapped.V[i]

        used_points = 0
        for i in range(5):
            used_points += abs(regions[i])

        assert used_points <= points

        act = np.append(regions, extra_points)
        for i in range(5): act[i] -= self.unwrapped.Amin[i]
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
              0,  # 9 Action points for next round
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
            40,  # 8 Round (Increased for 30+ safety)
            36,  # 9 Action points for next round
        ])
        self.observation_space = gym.spaces.Box(self.low, self.high)

    def observation(self, obs):

        new_obs = obs + self.env.unwrapped.Vmin

        return new_obs


class OekoPerRoundRewardWrapper(gym.Wrapper):
    def __init__(self, env, per_round_reward=1):
        super().__init__(env)
        self.per_round_reward = per_round_reward

    def mod_reward(self):
        if self.unwrapped.done and self.unwrapped.V[self.unwrapped.ROUND] in range(10, 31):
            reward = self.unwrapped.balance
        else:
            reward = self.per_round_reward
        return reward

    def step(self, action):
        obs, _, terminated, truncated, d = self.env.step(action)
        reward = self.mod_reward()
        return obs, reward, terminated, truncated, d


class OekoAuxRewardWrapper(gym.Wrapper):
    def __init__(self, env, scaling=1):
        super().__init__(env)
        self.scaling = scaling

    def mod_reward(self):
        if self.done and self.unwrapped.V[self.ROUND] in range(10, 31):
            return self.unwrapped.balance
        else:
            production_reward = 14 - abs(15 - self.unwrapped.V[self.PRODUCTION])
            population_reward = 23 - abs(24 - self.unwrapped.V[self.POPULATION])
            return self.scaling * (production_reward + population_reward)

    def step(self, action):
        obs, _, terminated, truncated, d = self.env.step(action)
        reward = self.mod_reward()
        return obs, reward, terminated, truncated, d


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
        regions_act = self.distribute1(regions_act, self.unwrapped.V[self.unwrapped.POINTS])
        if reduce_production:
            regions_act[1] = -regions_act[1]

        act = np.append(regions_act, special_act)
        act -= self.unwrapped.Amin
        act = np.clip(act, np.zeros(len(act)), self.unwrapped.Amax-self.unwrapped.Amin)
        return act


class OekoActionBuilderWrapper(gym.ActionWrapper):
    """
    This wrapper builds up an action by spending one point at a time and performs a step
    in the underlying environment once action 0 (move to next round) is chosen.
    Derived from gymcts-games for sequential tree search support.
    """

    def _calc_additional_population_points(self):
        v = getattr(self.env.unwrapped, 'V', self.env.unwrapped.init_v)
        education_level = v[self.env.unwrapped.EDUCATION]
        if education_level in range(21, 24):
            return 3
        elif education_level in range(24, 28):
            return 4
        elif education_level in range(28, 30):
            return 5
        else:
            return 0

    def _reset_wrapper_state(self):
        self._current_action_dict = {
            "Sanitation": 0,
            "Production": 0,
            "Education": 0,
            "Quality of Life": 0,
            "Population Growth": 0,
            "Population Growth extra": 0,
        }
        self._available_action_points = getattr(self.env.unwrapped, 'V', self.env.unwrapped.init_v)[self.env.unwrapped.POINTS]
        self._available_extra_points = self._calc_additional_population_points()

        self._production_change_direction = None  # None, "up", "down".
        self._population_extra_change_direction = None  # None, "up", "down".
        self._cached_obs = None

    def __init__(self, env):
        super().__init__(env)

        self._reset_wrapper_state()
        
        # Action space: 0: Next Round, 1: San+, 2: Prod+, 3: Prod-, 4: Edu+, 5: QoL+, 6: PG+, 7: PG_extra+, 8: PG_extra-
        self.action_space = gym.spaces.Discrete(9)
        
        # Extended observation space including buffered action values
        self.observation_space = gym.spaces.MultiDiscrete(
            list(self.env.observation_space.nvec) + [29, 59, 29, 29, 29, 11]
        )

    def valid_action_mask(self):
        next_round_valid = True
        
        env_pop_growth = self.env.unwrapped.V[self.env.unwrapped.POPULATION_GROWTH] + \
                         self._current_action_dict['Population Growth'] + \
                         self._current_action_dict['Population Growth extra']
        env_pop_growth_max = self.env.unwrapped.Vmax[self.env.unwrapped.POPULATION_GROWTH]
        env_pop_growth_min = self.env.unwrapped.Vmin[self.env.unwrapped.POPULATION_GROWTH]

        increase_extra_valid = self._available_extra_points > 0 and \
                               (self._population_extra_change_direction in ["up", None]) and \
                               (env_pop_growth + 1 <= env_pop_growth_max)
        
        decrease_extra_valid = self._available_extra_points > 0 and \
                               (self._population_extra_change_direction in ["down", None]) and \
                               (env_pop_growth - 1 >= env_pop_growth_min)

        if self._available_action_points <= 0:
            mask = np.zeros(9, dtype=bool)
            mask[0] = next_round_valid
            mask[7] = increase_extra_valid
            mask[8] = decrease_extra_valid
            return mask

        # Basic sector validations
        inc_san_valid = (self.env.unwrapped.V[0] + self._current_action_dict['Sanitation'] + 1) <= self.env.unwrapped.Vmax[0]
        inc_prod_valid = (self.env.unwrapped.V[1] + self._current_action_dict['Production'] + 1) <= self.env.unwrapped.Vmax[1]
        dec_prod_valid = (self.env.unwrapped.V[1] + self._current_action_dict['Production'] - 1) >= self.env.unwrapped.Vmin[1]
        inc_edu_valid = (self.env.unwrapped.V[2] + self._current_action_dict['Education'] + 1) <= self.env.unwrapped.Vmax[2]
        inc_qol_valid = (self.env.unwrapped.V[3] + self._current_action_dict['Quality of Life'] + 1) <= self.env.unwrapped.Vmax[3]
        inc_pg_valid = (self.env.unwrapped.V[4] + self._current_action_dict['Population Growth'] + 1) <= self.env.unwrapped.Vmax[4]

        return np.array([
            next_round_valid, inc_san_valid, inc_prod_valid, dec_prod_valid,
            inc_edu_valid, inc_qol_valid, inc_pg_valid,
            increase_extra_valid, decrease_extra_valid
        ])

    def action(self, action):
        # This is a bit tricky because gymnasium ActionWrapper expects us to return the action
        # for the underlying environment. But here we are building it sequentially.
        # We handle the state update in a separate method or within this one by intercepting.
        # However, for RL, we usually want a step() that returns immediately if it's just an internal update.
        # The gymcts implementation override step. Let's do that instead.
        return action # Dummy, see step() override below

    def step(self, action):
        mask = self.valid_action_mask()
        if not mask[action]:
            # Invalid action penalty or just return current state
            return self._extend_obs(self._cached_obs if self._cached_obs is not None else self.env.unwrapped.obs), -1.0, False, False, {"invalid": True}

        if action == 0:
            # Move to next round
            act_to_pass = np.array([
                self._current_action_dict["Sanitation"],
                self._current_action_dict["Production"],
                self._current_action_dict["Education"],
                self._current_action_dict["Quality of Life"],
                self._current_action_dict["Population Growth"],
                self._current_action_dict["Population Growth extra"],
            ])
            # The underlying environment adds self.Amin, so we must compensate
            act_to_pass -= self.env.unwrapped.Amin
            
            obs, reward, terminated, truncated, info = self.env.step(act_to_pass)
            self._reset_wrapper_state()
            self._cached_obs = obs
            return self._extend_obs(obs), reward, terminated, truncated, info

        # Internal state updates for sequential building
        if action == 1: self._current_action_dict["Sanitation"] += 1
        elif action == 2: 
            self._current_action_dict["Production"] += 1
            self._production_change_direction = "up"
        elif action == 3: 
            self._current_action_dict["Production"] -= 1
            self._production_change_direction = "down"
        elif action == 4: self._current_action_dict["Education"] += 1
        elif action == 5: self._current_action_dict["Quality of Life"] += 1
        elif action == 6: self._current_action_dict["Population Growth"] += 1
        elif action == 7: 
            self._current_action_dict["Population Growth extra"] += 1
            self._available_extra_points -= 1
            self._population_extra_change_direction = "up"
        elif action == 8: 
            self._current_action_dict["Population Growth extra"] -= 1
            self._available_extra_points -= 1
            self._population_extra_change_direction = "down"

        if action != 0:
            self._available_action_points -= 1 if action < 7 else 0

        # Return a small per-allocation reward if configured, otherwise 0
        return self._extend_obs(self._cached_obs if self._cached_obs is not None else self.env.unwrapped.obs), 0.0, False, False, {}

    def _extend_obs(self, obs):
        if obs is None:
            obs = self.env.unwrapped.obs
        buffered = np.array([
            self._current_action_dict["Sanitation"],
            self._current_action_dict["Production"] + 28,
            self._current_action_dict["Education"],
            self._current_action_dict["Quality of Life"],
            self._current_action_dict["Population Growth"],
            self._current_action_dict["Population Growth extra"] + 5
        ])
        return np.concatenate([obs, buffered])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_wrapper_state()
        self._cached_obs = obs.copy()
        return self._extend_obs(obs), info


class DynamicReserveGovernorV3(gym.ActionWrapper):
    """
    Governor V3: Ensures Maintenance AP is funded before Growth AP.
    Prevents systemic collapse by prioritizing QoL and Politics.
    """
    def __init__(self, env, safe_threshold=12.0):
        super().__init__(env)
        self.safe_threshold = safe_threshold

    def action(self, raw_action):
        """
        Intercepts raw actions and enforces maintenance reserves.
        raw_action: np.array([san, prod, edu, qol, pg, pg_extra])
        """
        v = self.env.unwrapped.V
        ap_total = v[9]
        
        # 1. Estimate deficits (Target 12.0 for stability corridor edge)
        qol_val = v[3]
        pol_val = v[7]
        
        # Heuristic: cost to restore to threshold (1 AP approx 1 Point)
        qol_deficit = max(0.0, self.safe_threshold - qol_val)
        pol_deficit = max(0.0, self.safe_threshold - pol_val)
        c_maint = qol_deficit + pol_deficit
        
        # 2. Partition AP
        if ap_total <= c_maint:
            # Crisis Mode: Divert all to QoL and Sanitation (Politics helper)
            safe_action = np.zeros_like(raw_action)
            if c_maint > 0:
                qol_ratio = qol_deficit / c_maint
                safe_action[3] = ap_total * qol_ratio  # QoL
                safe_action[0] = ap_total * (1 - qol_ratio)  # Sanitation
            return safe_action
        
        # 3. Growth Mode: Fund maintenance, then distribute remainder
        ap_growth = ap_total - c_maint
        
        # Start with maintenance
        safe_action = np.zeros_like(raw_action)
        safe_action[3] = qol_deficit # QoL
        safe_action[0] = pol_deficit # Use sanitation as proxy for politics stability
        
        # Normalize and distribute growth surplus based on agent policy
        # Filter indices for growth (Production=1, Education=2)
        growth_indices = [1, 2]
        raw_growth = np.abs(raw_action[growth_indices])
        total_growth_desire = np.sum(raw_growth) + 1e-8
        
        for i, idx in enumerate(growth_indices):
            safe_action[idx] += ap_growth * (raw_growth[i] / total_growth_desire)
            
        return safe_action


class HomeostaticRewardV3(gym.Wrapper):
    """
    Reward V3: Homeostatic Drive Reduction.
    Rewards reducing the distance to the "Patient Gardener" setpoint (16.0).
    """
    def __init__(self, env, target=16.0, exponent=2.0):
        super().__init__(env)
        self.target = target
        self.exponent = exponent
        self.prev_drive = None
        self.base_weights = np.array([1.0, 1.2, 1.5, 2.0, 2.0, 1.5, 1.0, 2.0])

    def _get_drive(self):
        state = self.env.unwrapped.V[:8]
        year = self.env.unwrapped.V[8]
        
        weights = np.copy(self.base_weights)
        m = self.exponent
        
        # Curriculum weights (from Research Report)
        if year <= 5:
            # Phase 1: Stabilization
            weights[3] *= 2.0 # QoL
            weights[7] *= 2.0 # Politics
            m = 1.5
        elif 6 <= year <= 14:
            # Phase 2: Transition (The Year 12 push)
            weights[2] *= 3.0 # Massive weight on Education (Education 21)
            weights[4] *= 2.5 # Heavy penalty for Pop Growth
            m = 2.0
        else:
            # Phase 3: Deep Homeostasis
            m = 2.5 # Strict adherence to setpoint
            
        distances = np.abs(state - self.target)
        return np.sum(weights * (distances ** m))

    def step(self, action):
        if self.prev_drive is None:
            self.prev_drive = self._get_drive()
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        curr_drive = self._get_drive()
        reduction = self.prev_drive - curr_drive
        
        # Survival length weighting
        year = self.env.unwrapped.V[8]
        
        if terminated and year < 30:
            reward = -20000.0
        elif terminated and year >= 30:
            reward = 50000.0
        else:
            reward = reduction
            
        self.prev_drive = curr_drive
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_drive = None
        return self.env.reset(**kwargs)
