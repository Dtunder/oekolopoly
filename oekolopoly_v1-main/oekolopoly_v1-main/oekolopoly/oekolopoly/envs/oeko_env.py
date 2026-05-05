import oekolopoly.oekolopoly.envs.get_boxes as gb
import gym
from gym import spaces
import numpy as np
import pygame


class OekoEnv(gym.Env):

    SANITATION        = 0
    PRODUCTION        = 1
    EDUCATION         = 2
    QUALITY_OF_LIFE   = 3
    POPULATION_GROWTH = 4
    ENVIRONMENT       = 5
    POPULATION        = 6
    POLITICS          = 7
    ROUND             = 8
    POINTS            = 9

    V_NAMES = [
        "Sanitation",
        "Production",
        "Education",
        "Quality of Life",
        "Population Growth",
        "Environment",
        "Population",
        "Politics",
        "Round",
        "Points",
    ]

    ACT_NAMES = [
        'SANITATION',
        'PRODUCTION',
        'EDUCATION',
        'QUALITY OF LIFE',
        'POPULATION GROWTH',
        'EXTRA',
    ]

    OBS_NAMES = V_NAMES[:-2]

    def __init__(self, render_mode=None):

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.last_v = None
        self.init_v = np.array([
             1,  # 0 Sanitation
            12,  # 1 Production
             4,  # 2 Education
            10,  # 3 Quality of Life
            20,  # 4 Population Growth
            13,  # 5 Environment
            21,  # 6 Population
             0,  # 7 Politics
             0,  # 8 Round
             8,  # 9 Points
        ])

        #                      0   1   2   3   4   5   6    7   8   9
        #                      S   Pr  Ed  Q   PG  En  Pop  Pol R   AP
        self.Vmin = np.array([ 1,  1,  1,  1,  1,  1,  1, -10,  0,  0])
        self.Vmax = np.array([29, 29, 29, 29, 29, 29, 48,  37, 30, 36])

        self.Amin = np.array([ 0,-28,  0,  0,  0, -5])
        self.Amax = np.array([28, 28, 28, 28, 28,  5])

        self.action_space = spaces.MultiDiscrete([
            29,  # 0 Sanitation
            57,  # 1 Production
            29,  # 2 Education
            29,  # 3 Quality of Life
            29,  # 4 Population Growth
            11,  # 5 box9 Education > Population Growth
        ])

        self.observation_space = spaces.MultiDiscrete([
            29,  # 0 Sanitation
            29,  # 1 Production
            29,  # 2 Education
            29,  # 3 Quality of Life
            29,  # 4 Population Growth
            29,  # 5 Environment
            48,  # 6 Population
            48,  # 7 Politics
            31,  # 8 Round
            37,  # 9 Actionpoints for next round
        ])

        self.done = False
        self.done_info = ''


    def seed(self, seed):
        """Dummy method needed by stable baselines3 when passing seed to model.
        The Oekolopoly environment is entirely deterministic"""
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
        if self.clock is not None:
            self.clock = None

    def render(self):
        if self.window is None and self.render_mode == "human":
            self.viewer_w = 500
            self.viewer_h = 500
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.viewer_w, self.viewer_h))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.viewer_w, self.viewer_h))
        canvas.fill((255, 255, 255))

        t      = 0.0
        t_step = 0.1  # speed of animation
        t_end  = 3.0

        print("Action:", self.curr_action)
        if self.done: print("Done:", self.done_info)

        while t < t_end:
            self.render_rounds(canvas,  5, 485, min(t, 1.0))  # progress bar, shows number of round
            self.render_action(canvas, 50,  70, min(t, 1.0))  # 2nd bar chart, A (positionHorizontal, positionVertical, time)
            self.render_vector(canvas, 50, 200, min(t, 1.0))  # 1st bar chart, V

            if self.render_mode == "human":
                # The following line copies our drawings from `canvas` to the visible window
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
                self.clock.tick(25)

            t += t_step

    def render_rounds(self, canvas, x_offset, y_offset, t):
        v0 = self.prev_result[self.ROUND]
        v1 = self.curr_result[self.ROUND]
        v  = v0 + (v1 - v0) * t  # t = 1 show final result, t = 0 show first result, t = 0.5 sth inbetween

        x = x_offset
        y = y_offset
        w = 490
        h = 10

        pygame.draw.rect(
            canvas,
            (int(255*0.94), int(255*0.94), int(255*0.94)),  # gray
            pygame.Rect(x, y, w, h)
        )

        w = int(v / self.Vmax[self.ROUND] * w)

        pygame.draw.rect(
            canvas,
            (int(255*0.6), int(255*0.6), int(255*0.6)),  # dark gray
            pygame.Rect(x, y, w, h)
        )

    def render_action(self, canvas, x_offset, y_offset, t):
        w = 40
        x_stride = 50  # distance between start of one to middle of another, distance = 50 - 40
        h_step = 8     # scale

        bg_x = x_offset - 20
        bg_y = y_offset
        bg_w = x_stride * len(self.ACT_NAMES) + 20 * 2
        bg_h = 10 * h_step

        pygame.draw.rect(
            canvas,
            (int(255*0.94), int(255*0.94), int(255*0.94)),
            pygame.Rect(bg_x, bg_y, bg_w, bg_h)
        )

        for i in range(len(self.ACT_NAMES)):
            x = i * x_stride + x_offset
            y = y_offset

            v0 = self.prev_action[i]
            v1 = self.curr_action[i]
            v  = v0 + (v1 - v0) * t
            h  = int(v * h_step)

            pygame.draw.rect(
                canvas,
                (int(255*0.2), int(255*0.2), int(255*1.)),
                pygame.Rect(x, y, w, h)
            )

    def render_vector(self, canvas, x_offset, y_offset, t):
        w = 40
        x_stride = 50
        h_step = 8

        bg_x = x_offset - 20
        bg_y = y_offset
        bg_w = x_stride * len(self.OBS_NAMES) + 20 * 2
        bg_h = 29 * h_step

        pygame.draw.rect(
            canvas,
            (int(255*0.94), int(255*0.94), int(255*0.94)),
            pygame.Rect(bg_x, bg_y, bg_w, bg_h)
        )

        for i in range(len(self.OBS_NAMES)):
            x = i * x_stride + x_offset
            y = y_offset

            v0 = self.prev_result[i]
            v1 = self.curr_result[i]
            v  = v0 + (v1 - v0) * t
            h  = int(v * h_step)

            r = (v - self.Vmin[i]) / (self.Vmax[i] - self.Vmin[i])
            r = abs((r - 0.5) * 2)
            g = 1 - r

            pygame.draw.rect(
                canvas,
                (int(255*r), int(255*g), 0),
                pygame.Rect(x, y, w, h)
            )

    def update_values(self, action):

        OOR = " out of allowed range"
        done = False
        done_info = None
        extra_points = action[5]

        # Update V and boxes
        box1 = gb.get_box1(self.V[self.SANITATION])
        if not done:
            self.V[self.ENVIRONMENT] += box1
            if self.V[self.ENVIRONMENT] not in range(1, 30):
                done = True
                if self.V[self.ENVIRONMENT] > 29: l = "high. "
                else: l = "low. "
                done_info = "Environment too " + l + str(self.V[self.ENVIRONMENT]) + OOR

        if not done:
            box2 = gb.get_box2(self.V[self.SANITATION])
            self.V[self.SANITATION] += box2
            if self.V[self.SANITATION] not in range(1, 30):
                done = True
                if self.V[self.SANITATION] > 29: l = "high. "
                else: l = "low. "
                done_info = "Sanitation too " + l + str(self.V[self.SANITATION]) + OOR

        if not done:
            box3 = gb.get_box3(self.V[self.PRODUCTION])
            self.V[self.PRODUCTION] += box3
            if self.V[self.PRODUCTION] not in range(1, 30):
                done = True
                if self.V[self.PRODUCTION] > 29: l = "high. "
                else: l = "low. "
                done_info = "Production too " + l + str(self.V[self.PRODUCTION]) + OOR

        if not done:
            box4 = gb.get_box4(self.V[self.PRODUCTION])
            self.V[self.ENVIRONMENT] += box4
            if self.V[self.ENVIRONMENT] not in range(1, 30):
                done = True
                if self.V[self.ENVIRONMENT] > 29: l = "high. "
                else: l = "low. "
                done_info = "Environment too " + l + str(self.V[self.ENVIRONMENT]) + OOR

        if not done:
            box5 = gb.get_box5(self.V[self.ENVIRONMENT])
            self.V[self.ENVIRONMENT] += box5
            if self.V[self.ENVIRONMENT] not in range(1, 30):
                done = True
                if self.V[self.ENVIRONMENT] > 29: l = "high. "
                else: l = "low. "
                done_info = "Environment too " + l + str(self.V[self.ENVIRONMENT]) + OOR

        if not done:
            box6 = gb.get_box6(self.V[self.ENVIRONMENT])
            self.V[self.QUALITY_OF_LIFE] += box6
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                if self.V[self.QUALITY_OF_LIFE] > 29: l = "high. "
                else: l = "low. "
                done_info = "Quality of Life too " + l + str(self.V[self.QUALITY_OF_LIFE]) + OOR

        if not done:
            box7 = gb.get_box7(self.V[self.EDUCATION])
            self.V[self.EDUCATION] += box7
            if self.V[self.EDUCATION] not in range(1, 30):
                done = True
                if self.V[self.EDUCATION] > 29: l = "high. "
                else: l = "low. "
                done_info = "Education too " + l + str(self.V[self.EDUCATION]) + OOR

        if not done:
            box8 = gb.get_box8(self.V[self.EDUCATION])
            self.V[self.QUALITY_OF_LIFE] += box8
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                if self.V[self.QUALITY_OF_LIFE] > 29: l = "high. "
                else: l = "low. "
                done_info = "Quality of Life too " + l + str(self.V[self.QUALITY_OF_LIFE]) + OOR

        if not done:
            if self.V[self.EDUCATION] in range(21, 24): extra_points = max(-3, min(3, extra_points))
            if self.V[self.EDUCATION] in range(24, 28): extra_points = max(-4, min(4, extra_points))
            if self.V[self.EDUCATION] in range(28, 30): extra_points = max(-5, min(5, extra_points))
            if self.V[self.EDUCATION] < 21: extra_points = 0
            box9 = gb.get_box9(self.V[self.EDUCATION], extra_points)
            self.V[self.POPULATION_GROWTH] += box9
            if self.V[self.POPULATION_GROWTH] not in range(1, 30):
                done = True
                if self.V[self.POPULATION_GROWTH] > 29: l = "high. "
                else: l = "low. "
                done_info = "Population Growth too " + l + str(self.V[self.POPULATION_GROWTH]) + OOR

        if not done:
            box10 = gb.get_box10(self.V[self.QUALITY_OF_LIFE])
            self.V[self.QUALITY_OF_LIFE] += box10
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                if self.V[self.QUALITY_OF_LIFE] > 29: l = "high. "
                else: l = "low. "
                done_info = "Quality of Life too " + l + str(self.V[self.QUALITY_OF_LIFE]) + OOR

        if not done:
            box11 = gb.get_box11(self.V[self.QUALITY_OF_LIFE])
            self.V[self.POPULATION_GROWTH] += box11
            if self.V[self.POPULATION_GROWTH] not in range(1, 30):
                done = True
                if self.V[self.POPULATION_GROWTH] > 29: l = "high. "
                else: l = "low. "
                done_info = "Population Growth too " + l + str(self.V[self.POPULATION_GROWTH]) + OOR

        if not done:
            box12 = gb.get_box12(self.V[self.QUALITY_OF_LIFE])
            self.V[self.POLITICS] += box12
            if self.V[self.POLITICS] not in range(-10, 38):
                done = True
                if self.V[self.POLITICS] > 37: l = "high. "
                else: l = "low. "
                done_info = "Politics too " + l + str(self.V[self.POLITICS]) + OOR

        if not done:
            box13 = gb.get_box13(self.V[self.POPULATION_GROWTH])
            boxW  = gb.get_boxW (self.V[self.POPULATION])
            self.V[self.POPULATION] += box13 * boxW
            if self.V[self.POPULATION] not in range(1, 49):
                done = True
                if self.V[self.POPULATION] > 48: l = "high. "
                else: l = "low. "
                done_info = "Population too " + l + str(self.V[self.POPULATION]) + OOR

        if not done:
            box14 = gb.get_box14(self.V[self.POPULATION])
            self.V[self.QUALITY_OF_LIFE] += box14
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                if self.V[self.QUALITY_OF_LIFE] > 29: l = "high. "
                else: l = "low. "
                done_info = "Quality of Life too " + l + str(self.V[self.QUALITY_OF_LIFE]) + OOR

        return self.V, done, done_info

    def step(self, action):
        clipping = True
        return self.__inner_step(action, clipping)

    def step_w_o_clip(self, action):
        clipping = False
        return self.__inner_step(action, clipping)

    def __inner_step(self, action, clipping=True):
        """ realized as inner class because step(self,action) does not allow extra arguments """

        assert self.action_space.contains(action), f"Action not in action_space: {action}"

        # Transform action space
        action = action + self.Amin
        self.prev_action = self.curr_action
        self.curr_action = action.copy()

        # Init
        self.done = False
        used_points = 0

        # Sum points from action
        used_points += action[self.SANITATION]
        used_points += abs(action[self.PRODUCTION])
        used_points += action[self.EDUCATION]
        used_points += action[self.QUALITY_OF_LIFE]
        used_points += action[self.POPULATION_GROWTH]

        if used_points < 0 or used_points > self.V[self.POINTS]:
            self.done = True
            if used_points < 0:
                done_reason = "Tried to use negative amount of actionpoints. "
            elif used_points > self.V[self.POINTS]:
                done_reason = "Tried to exceed available amount of actionpoints. "
            done_reason += f"Tried to use {used_points} action points, but only between 0 and {self.V[self.POINTS]} are available"
            return self.obs, 0, self.done, {'balance (always)': self.balance_always,
                                            'balance_numerator (always)': self.balance_numerator_always,
                                            'balance': self.balance,
                                            'balance_numerator': self.balance_numerator,
                                            'round': self.V[self.ROUND],
                                            'done_reason': done_reason,
                                            'valid_move': False,
                                            'invalid_move_info': "Unavailable number of actionpoints were used in this round."}
        assert 0 <= used_points <= self.V[self.POINTS], f"Action takes too many points: action={action} POINTS={self.V[self.POINTS]})"

        for i in range(5):
            if self.V[i] + action[i] not in range(self.Vmin[i], self.Vmax[i] + 1):
                self.done = True
                if self.V[i] + action[i] < self.Vmin[i]:
                    done_reason = f"Distribution of actionpoints pushes {self.V_NAMES[i]} below limit. "
                elif self.V[i] + action[i] > self.Vmax[i]:
                    done_reason = f"Distribution of actionpoints pushes {self.V_NAMES[i]} above limit. "
                done_reason += f"Tried to use {self.V[i] + action[i]} action points for {self.V_NAMES[i]}, but only between {self.Vmin[i]} and {self.Vmax[i]} are available"
                return self.obs, 0, self.done, {'balance (always)': self.balance_always,
                                                'balance_numerator (always)': self.balance_numerator_always,
                                                'balance': self.balance,
                                                'balance_numerator': self.balance_numerator,
                                                'round': self.V[self.ROUND],
                                                'done_reason': done_reason,
                                                'valid_move': False,
                                                'invalid_move_info': f"Unavailable number of actionpoints assigned to {self.V_NAMES[i]}."}
            assert (self.V[i] + action[i]) in range(self.Vmin[i], self.Vmax[i] + 1), f"Action puts region out of action[{i}]: action={action} V={self.V}"

        # The turn is valid

        for i in range(5): self.V[i] += action[i]

        # Update boxes and V accordingly
        self.V, self.done, self.done_info = self.update_values(action)

        # Update points and round
        self.V[self.POINTS] -= used_points
        self.V[self.ROUND]  += 1

        # Clip values if not in range
        if clipping:
            for i in range(8):
                if self.V[i] not in range(self.Vmin[i], self.Vmax[i] + 1):
                    self.V[i] = max(self.Vmin[i], min(self.Vmax[i], self.V[i]))
                    self.done = True

        if self.V[self.ROUND] == 30:
            self.done = True
            self.done_info = 'Maximum number of rounds reached.'

        # Points for next round
        if self.done:
            self.V[self.POINTS] = 0
        else:
            boxA = gb.get_boxA(self.V[self.POPULATION])
            boxB = gb.get_boxB(self.V[self.POLITICS])
            boxC = gb.get_boxC(self.V[self.PRODUCTION])
            boxV = gb.get_boxV(self.V[self.PRODUCTION])
            boxD = gb.get_boxD(self.V[self.QUALITY_OF_LIFE])

            self.V[self.POINTS] += boxA * boxV
            self.V[self.POINTS] += boxB
            self.V[self.POINTS] += boxC
            self.V[self.POINTS] += boxD

        if self.V[self.POINTS] < 0:
            self.V[self.POINTS] = 0
            self.done = True
            self.done_info = 'Minimum amount of actionpoints reached.'

        if self.V[self.POINTS] > 36:
            self.V[self.POINTS] = 36
            self.done = True
            self.done_info = 'Maximum number of actionpoints reached.'

        boxD = gb.get_boxD(self.V[self.QUALITY_OF_LIFE])
        a = float((boxD * 3 + self.V[self.POLITICS]) * 10)
        b = float(self.V[self.ROUND] + 3)
        self.balance_numerator_always = int(a)
        self.balance_always = a / b


        # Transform V in obs
        self.obs = self.V - self.Vmin
        if clipping:
            assert self.observation_space.contains(self.obs), f"obs not in observation_space: obs={self.obs}"

        if self.V[self.ROUND] in range(10, 31):
            self.balance = self.balance_always
            self.balance_numerator = self.balance_numerator_always
        else:
            self.balance = 0
            self.balance_numerator = 0

        if self.done and self.V[self.ROUND] in range(10, 31):
            reward = self.balance
        else:
            reward = 0

        self.prev_result = self.curr_result
        self.curr_result = self.V.copy()

        self.last_v    = self.V.copy()

        if self.render_mode == "human":
            self.render()

        return self.obs, reward, self.done, {'balance (always)': self.balance_always,
                                            'balance_numerator (always)': self.balance_numerator_always,
                                            'balance': self.balance,
                                            'balance_numerator': self.balance_numerator,
                                            'round': self.V[self.ROUND],
                                            'done_reason': self.done_info,
                                            'valid_move': True,
                                            'invalid_move_info': ''}

    def get_initial_v(self):
        return self.init_v.copy()

    def set_v(self, init_v):
        self.init_v = init_v

    def reset(self, options=None):
        if options is not None and "v" in options:
            self.V = np.array(options["v"])  # non-default initial values v
        else:
            self.V = self.get_initial_v()

        self.curr_action = np.zeros(self.action_space.shape[0], 'int64')
        self.curr_result = self.V.copy()

        self.done = False
        self.done_info = ''

        boxD = gb.get_boxD(self.V[self.QUALITY_OF_LIFE])
        a = float((boxD * 3 + self.V[self.POLITICS]) * 10)
        b = float(self.V[self.ROUND] + 3)
        self.balance_numerator_always = int(a)
        self.balance_always = a / b
        self.balance = 0
        self.balance_numerator = 0

        self.obs = self.V - self.Vmin
        assert self.observation_space.contains(self.obs), "obs not in observation_space"

        return self.obs
