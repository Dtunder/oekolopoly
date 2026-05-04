# import oekolopoly.oekolopoly.envs.get_boxes as gb
from oekolopoly.oekolopoly.envs.get_boxes2 import GetBoxes as gb
import gymnasium as gym
from gymnasium import spaces
import numpy as np
try:
    import pygame
except ImportError:
    pygame = None

from translator import dict_translate, env_translate


class OekoEnv(gym.Env):
    """
        This class realizes the environment for the game Ökolopoly.

        The environment is based on package `gymnasium==1.2.2` as `gym`.

        Method `step` returns the five elements `obs, reward, terminated, truncated, info` where
        always `truncated=False` and thus `terminated` is equivalent to `done`.
        `info` is a dict with the following key-value pairs:

        - 'balance (always)': the game balance according to the Ökolopoly balance formula, irrespective of round
        - 'balance_numerator (always)': the numerator of this formula
        - 'balance': `info['balance (always)']` if round in [10,30], 0 else
        - 'balance_numerator': `info['balance_numerator (always)']` if round in [10,30], 0 else
        - 'round': the number of rounds played
        - 'done_reason': string with the reason for termination (`None` if not yet terminated)
        - 'done_reason_detail': string with further info
        - 'valid_move': `False` if the action points were distributed in an unallowed way, `True` else
        - 'invalid_move_info': string describing the reason for invalid move

    """
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

    def __init__(self, render_mode=None, language="en"):

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.viewer_w = None
        self.viewer_h = None
        self.dtl = dtl = dict_translate(language)
        self.etl = etl = env_translate(language)
        # print("TEST DTL: " + dtl["APoints"])

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
        ], dtype=np.int32)

        #                      0   1   2   3   4   5   6    7   8   9
        #                      S  Pr  Ed  Q   PG  En  Pop  Pol  R  AP
        self.Vmin = np.array([ 1,  1,  1,  1,  1,  1,  1, -10,  0,  0], dtype=np.int32)
        self.Vmax = np.array([29, 29, 29, 29, 29, 29, 48,  37, 30, 36], dtype=np.int32)

        #                      0   1   2   3   4   5
        #                      S  Pr  Ed   Q  PG  SC
        self.Amin = np.array([ 0,-28,  0,  0,  0, -5], dtype=np.int32)
        self.Amax = np.array([28, 28, 28, 28, 28,  5], dtype=np.int32)

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
            37,  # 9 Action points for next round
        ])

        self.done = False
        self.done_info = ''
        self.done_reason_detail = ''
        self.info = dict()

    # --- should be obsolete after migrating to gymnasium ---
    # def seed(self, seed):
    #     """Dummy method needed by stable baselines3 when passing seed to model.
    #     The Oekolopoly environment is entirely deterministic"""
    #     pass

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
        if self.done: print(f"Done: {self.done_info}{self.done_reason_detail}")

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

        OOR = self.etl[" out of allowed range"]
        done = False
        done_info = None
        done_reason_detail = None
        extra_points = action[5]

        # Update V and boxes
        box1 = gb.get_box1(self.clip(self.SANITATION))
        if not done:
            self.V[self.ENVIRONMENT] += box1
            if self.V[self.ENVIRONMENT] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.ENVIRONMENT] > 29 else self.etl[" too low. "]
                done_info = self.dtl["EnvirDamage"] + l
                done_reason_detail = f"{self.V[self.ENVIRONMENT]} {OOR} (1, ..., 29)."

        if not done:
            box2 = gb.get_box2(self.clip(self.SANITATION))
            self.V[self.SANITATION] += box2
            if self.V[self.SANITATION] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.SANITATION] > 29 else self.etl[" too low. "]
                done_info = self.dtl["Redevelop"] + l
                done_reason_detail = f"{self.V[self.SANITATION]} {OOR} (1, ..., 29)."

        if not done:
            box3 = gb.get_box3(self.clip(self.PRODUCTION))
            self.V[self.PRODUCTION] += box3
            if self.V[self.PRODUCTION] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.PRODUCTION] > 29 else self.etl[" too low. "]
                done_info = self.dtl["Production"] + l
                done_reason_detail = f"{self.V[self.SANITATION]} {OOR} (1, ..., 29)."

        if not done:
            box4 = gb.get_box4(self.clip(self.PRODUCTION))
            self.V[self.ENVIRONMENT] += box4
            if self.V[self.ENVIRONMENT] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.ENVIRONMENT] > 29 else self.etl[" too low. "]
                done_info = self.dtl["EnvirDamage"] + l
                done_reason_detail = f"{self.V[self.ENVIRONMENT]} {OOR} (1, ..., 29)."

        if not done:
            box5 = gb.get_box5(self.clip(self.ENVIRONMENT))
            self.V[self.ENVIRONMENT] += box5
            if self.V[self.ENVIRONMENT] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.ENVIRONMENT] > 29 else self.etl[" too low. "]
                done_info = self.dtl["EnvirDamage"] + l
                done_reason_detail = f"{self.V[self.ENVIRONMENT]} {OOR} (1, ..., 29)."

        if not done:
            box6 = gb.get_box6(self.clip(self.ENVIRONMENT))
            self.V[self.QUALITY_OF_LIFE] += box6
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.QUALITY_OF_LIFE] > 29 else self.etl[" too low. "]
                done_info = self.dtl["QualityOfLife"] + l
                done_reason_detail = f"{self.V[self.QUALITY_OF_LIFE]} {OOR} (1, ..., 29)."

        if not done:
            box7 = gb.get_box7(self.clip(self.EDUCATION))
            self.V[self.EDUCATION] += box7
            if self.V[self.EDUCATION] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.EDUCATION] > 29 else self.etl[" too low. "]
                done_info = self.dtl["Enlightenment"] + l
                done_reason_detail = f"{self.V[self.EDUCATION]} {OOR} (1, ..., 29)."

        if not done:
            box8 = gb.get_box8(self.clip(self.EDUCATION))
            self.V[self.QUALITY_OF_LIFE] += box8
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.QUALITY_OF_LIFE] > 29 else self.etl[" too low. "]
                done_info = self.dtl["QualityOfLife"] + l
                done_reason_detail = f"{self.V[self.QUALITY_OF_LIFE]} {OOR} (1, ..., 29)."

        if not done:
            if   self.V[self.EDUCATION] in range(21, 24): extra_points = max(-3, min(3, extra_points))
            elif self.V[self.EDUCATION] in range(24, 28): extra_points = max(-4, min(4, extra_points))
            elif self.V[self.EDUCATION] in range(28, 30): extra_points = max(-5, min(5, extra_points))
            else:  # i.e. if self.V[self.EDUCATION] < 21:
                extra_points = 0
            box9 = gb.get_box9(self.V[self.EDUCATION], extra_points)
            self.V[self.POPULATION_GROWTH] += box9
            if self.V[self.POPULATION_GROWTH] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.POPULATION_GROWTH] > 29 else self.etl[" too low. "]
                done_info = self.dtl["ReproRate"] + l
                done_reason_detail = f"{self.V[self.POPULATION_GROWTH]} {OOR} (1, ..., 29)."

        if not done:
            box10 = gb.get_box10(self.clip(self.QUALITY_OF_LIFE))
            self.V[self.QUALITY_OF_LIFE] += box10
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.QUALITY_OF_LIFE] > 29 else self.etl[" too low. "]
                done_info = self.dtl["QualityOfLife"] + l
                done_reason_detail = f"{self.V[self.QUALITY_OF_LIFE]} {OOR} (1, ..., 29)."

        if not done:
            box11 = gb.get_box11(self.clip(self.QUALITY_OF_LIFE))
            self.V[self.POPULATION_GROWTH] += box11
            if self.V[self.POPULATION_GROWTH] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.POPULATION_GROWTH] > 29 else self.etl[" too low. "]
                done_info = self.dtl["ReproRate"] + l
                done_reason_detail = f"{self.V[self.POPULATION_GROWTH]} {OOR} (1, ..., 29)."

        if not done:
            box12 = gb.get_box12(self.clip(self.QUALITY_OF_LIFE))
            self.V[self.POLITICS] += box12
            if self.V[self.POLITICS] not in range(-10, 38):
                done = True
                l = self.etl[" too high. "] if self.V[self.POLITICS] > 37 else self.etl[" too low. "]
                done_info = self.dtl["Politics"] + l
                done_reason_detail = f"{self.V[self.POLITICS]} {OOR} (-10, ..., 37)."

        if not done:
            box13 = gb.get_box13(self.clip(self.POPULATION_GROWTH))
            boxW  = gb.get_boxW(self.clip(self.POPULATION))
            self.V[self.POPULATION] += box13 * boxW
            if self.V[self.POPULATION] not in range(1, 49):
                done = True
                l = self.etl[" too high. "] if self.V[self.POPULATION] > 48 else self.etl[" too low. "]
                done_info = self.dtl["Population"] + l
                done_reason_detail = f"{self.V[self.POPULATION]} {OOR} (1, ..., 48)."

        if not done:
            box14 = gb.get_box14(self.clip(self.POPULATION))
            self.V[self.QUALITY_OF_LIFE] += box14
            if self.V[self.QUALITY_OF_LIFE] not in range(1, 30):
                done = True
                l = self.etl[" too high. "] if self.V[self.QUALITY_OF_LIFE] > 29 else self.etl[" too low. "]
                done_info = self.dtl["QualityOfLife"] + l
                done_reason_detail = f"{self.V[self.QUALITY_OF_LIFE]} {OOR} (1, ..., 29)."

        return self.V, done, done_info, done_reason_detail

    def clip(self, s: int) -> np.int32:
        return np.clip(self.V[s], self.Vmin[s], self.Vmax[s])

    def step(self, action):
        clipping = True
        return self.__inner_step(action, clipping)

    def step_w_o_clip(self, action):
        clipping = False
        return self.__inner_step(action, clipping)

    def __inner_step(self, action, clipping=True):
        """
            Realized as inner class because step(self,action) does not allow extra arguments.

            Note that ``obs = self.V - self.Vmin``

        :param action:   the action to perform
        :param clipping: whether to clip the values of ``self.V`` or not
        :return:         ``obs, reward, terminated, truncated, info``
        """
        assert self.action_space.contains(action), f"Action not in action_space: {action}"

        # Transform action space
        action = action + self.Amin
        self.prev_action = self.curr_action
        self.curr_action = action.copy()

        # Init
        self.done = False
        truncated = False
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
                done_reason = "Tried to use negative amount of action points. "
            else:   # i.e. if used_points > self.V[self.POINTS]:
                done_reason = "Tried to exceed available amount of action points. "
            # done_reason += f"Tried to use {used_points} action points, but only between 0 and {self.V[self.POINTS]} are available"
            done_reason_detail = f"{used_points} not in (0, ..., {self.V[self.POINTS]})"
            return self.obs, 0, self.done, truncated, {'balance (always)': self.balance_always,
                                                       'balance_numerator (always)': self.balance_numerator_always,
                                                       'balance': self.balance,
                                                       'balance_numerator': self.balance_numerator,
                                                       'round': self.V[self.ROUND],
                                                       'done_reason': done_reason,
                                                       'done_reason_detail': done_reason_detail,
                                                       'valid_move': False,
                                                       'invalid_move_info': "Invalid number of action points"}
        assert 0 <= used_points <= self.V[self.POINTS], \
            f"Action takes too many points: action={action} POINTS={self.V[self.POINTS]})"

        for i in range(5):
            if self.V[i] + action[i] not in range(self.Vmin[i], self.Vmax[i] + 1):
                self.done = True
                if self.V[i] + action[i] < self.Vmin[i]:
                    done_reason = f"Distribution of action points pushes {self.V_NAMES[i]} below limit. "
                else:   # i.e. if self.V[i] + action[i] > self.Vmax[i]:
                    done_reason = f"Distribution of action points pushes {self.V_NAMES[i]} above limit. "
                done_reason_detail = f"{self.V[i] + action[i]} action points are not in ({self.Vmin[i]}, ..., {self.Vmax[i]})"
                inval_m_info = f"Invalid number of action points assigned to {self.V_NAMES[i]}."
                return self.obs, 0, self.done, truncated, {'balance (always)': self.balance_always,
                                                           'balance_numerator (always)': self.balance_numerator_always,
                                                           'balance': self.balance,
                                                           'balance_numerator': self.balance_numerator,
                                                           'round': self.V[self.ROUND],
                                                           'done_reason': done_reason,
                                                           'done_reason_detail': done_reason_detail,
                                                           'valid_move': False,
                                                           'invalid_move_info': inval_m_info,
                                                           }
            assert (self.V[i] + action[i]) in range(self.Vmin[i], self.Vmax[i] + 1), \
                    f"Action puts region out of action[{i}]: action={action} V={self.V}"

        # The turn is valid

        for i in range(5): self.V[i] += action[i]

        # Update boxes and V accordingly
        self.V, self.done, self.done_info, self.done_reason_detail = self.update_values(action)

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
            self.done_info = self.etl["MaxNumRoundsReached"]
            self.done_reason_detail = f" {self.dtl["Round"]} {self.V[self.ROUND]}"

        # Points for next round
        if self.done:
            self.V[self.POINTS] = 0
        else:
            boxA = gb.get_boxA(self.clip(self.POPULATION))
            boxB = gb.get_boxB(self.clip(self.POLITICS))
            boxC = gb.get_boxC(self.clip(self.PRODUCTION))
            boxV = gb.get_boxV(self.clip(self.PRODUCTION))
            boxD = gb.get_boxD(self.clip(self.QUALITY_OF_LIFE))

            self.V[self.POINTS] += boxA * boxV
            self.V[self.POINTS] += boxB
            self.V[self.POINTS] += boxC
            self.V[self.POINTS] += boxD

        OOR = self.etl[" out of allowed range"]
        if self.V[self.POINTS] < 0:
            self.done = True
            self.done_info = self.etl["NumAPointsTooLow"]+". "
            self.done_reason_detail = f"{self.V[self.POINTS]} {OOR} (0, ..., 36)."
            if clipping: self.V[self.POINTS] = 0            # /2025/12/11/WK/ Bug fix: added 'if clipping' to enforce
                                                            # that a step_w_o_clip (preview mode!) will not clip the
                                                            # action points and will display the correct value change
        if self.V[self.POINTS] > 36:
            self.done = True
            self.done_info = self.etl["NumAPointsTooHigh"]+". "
            self.done_reason_detail = f"{self.V[self.POINTS]} {OOR} (0, ..., 36)."
            if clipping: self.V[self.POINTS] = 36           # /2025/12/11/WK/ Bug fix: added 'if clipping' (see above)

        boxD = gb.get_boxD(self.clip(self.QUALITY_OF_LIFE))
        a = float((boxD * 3 + self.clip(self.POLITICS)) * 10)
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

        self.last_v = self.V.copy()

        if self.render_mode == "human":
            self.render()

        return self.obs, reward, self.done, truncated, {'balance (always)': self.balance_always,
                                                        'balance_numerator (always)': self.balance_numerator_always,
                                                        'balance': self.balance,
                                                        'balance_numerator': self.balance_numerator,
                                                        'round': self.V[self.ROUND],
                                                        'done_reason': self.done_info,
                                                        'done_reason_detail': self.done_reason_detail,
                                                        'valid_move': True,
                                                        'invalid_move_info': ''}

    def get_initial_v(self):
        return self.init_v.copy()

    def set_v(self, init_v):
        self.init_v = init_v

    def reset(self, options=None, seed=None):
        if options is not None and "v" in options:
            self.V = np.array(options["v"])  # non-default initial values v
        else:
            self.V = self.get_initial_v()

        self.curr_action = np.zeros(self.action_space.shape[0], 'int64')
        self.curr_result = self.V.copy()

        self.done = False
        self.done_info = ''
        self.done_reason_detail = ''

        boxD = gb.get_boxD(self.clip(self.QUALITY_OF_LIFE))
        a = float((boxD * 3 + self.V[self.POLITICS]) * 10)
        b = float(self.V[self.ROUND] + 3)
        self.balance_numerator_always = int(a)
        self.balance_always = a / b
        self.balance = 0
        self.balance_numerator = 0

        self.obs = self.V - self.Vmin
        assert self.observation_space.contains(self.obs), "obs not in observation_space"

        return self.obs, self.info
