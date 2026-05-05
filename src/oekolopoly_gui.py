# --------------------------------------------------------------------------------------------------------
#    Ökolopoly GUI created by Alexander Albers, Wolfgang Konen
#        German or English version depending on command line argument
#               python -m oekol_lang_gui --language "de"        (default) - or -
#               python -m oekol_lang_gui --language "en"
# --------------------------------------------------------------------------------------------------------
# import random
# import uuid
import argparse
import os
import sys
import gymnasium as gym
import logging
from typing import List, Tuple, Optional, Any
import copy
from pygame.math import Vector2
import numpy as np
import pygame

# Configure Professional Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SovereignChampion")

def lazy_load_torch():
    """Deferred loading of heavy AI libraries to prevent Windows/Python 3.14 hangs."""
    logger.info("Initializing Sovereign AI Engine...")
    import torch
    from sb3_contrib import RecurrentPPO
    import torch.nn as nn
    
    # WATERPROOF MONKEY PATCH for LSTM
    original_lstm_init = nn.LSTM.__init__
    def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
        return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
    nn.LSTM.__init__ = patched_lstm_init
    
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    return torch, RecurrentPPO

# Define Absolute Root Path for Bulletproof Execution
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NASUTA_ROOT = os.path.join(ROOT_DIR, "gymcts-games-main", "gymcts-games-main", "src")

if NASUTA_ROOT not in sys.path:
    sys.path.insert(0, NASUTA_ROOT)
if os.path.join(ROOT_DIR, "src") not in sys.path:
    sys.path.append(os.path.join(ROOT_DIR, "src"))

os.chdir(ROOT_DIR) # Force working directory to script location

# Module Bridge for Compatibility
import oekolopoly.env.oeko_env as oeko_env
sys.modules['oekolopoly.oekolopoly'] = oeko_env

# Manual Gym Registration
from gymnasium.envs.registration import register
try:
    register(
        id='Oekolopoly-v2',
        entry_point='oekolopoly.env.oeko_env:OekoEnv',
    )
except:
    pass

from translator import dict_translate, dict_help_screens
from mcts_planner import SovereignMCTS

class SovereignGuardian:
    def __init__(self, env):
        self.env = env

    def get_final_action(self, avail):
        V = self.env.unwrapped.V
        dist = np.zeros(5, dtype=int)
        reasons = []
        
        # 1. CORE INVESTMENTS
        if V[2] < 29 and avail > 0:
            d = min(avail, 29 - int(V[2])); dist[2] = int(d); avail -= dist[2]
            reasons.append(f"Core: Education yield (+{d}).")
            
        # 2. SURVIVAL TARGETS
        p_target = 13
        p_dist = p_target - int(V[1])
        if avail > 0:
            d = min(max(1, avail // 2), abs(p_dist))
            dist[1] = -int(d) if p_dist < 0 else int(d); avail -= abs(dist[1])
            reasons.append(f"Adaptive: Target Prod {p_target}.")
            
        # 3. ALCHEMIST BURN (Avoid 35)
        while avail + int(V[9]) > 28:
            changed = False
            if int(V[2]) + dist[2] < 29:
                dist[2] += 1; avail -= 1; changed = True
                reasons.append("Burn: Education.")
            elif int(V[3]) + dist[3] < 18:
                dist[3] += 1; avail -= 1; changed = True
                reasons.append("Burn: Raising QoL.")
            elif int(V[4]) + dist[4] < 15:
                dist[4] += 1; avail -= 1; changed = True
                reasons.append("Burn: Population.")
            elif V[5] < 12:
                if int(V[1]) + dist[1] < 18:
                    dist[1] += 1; avail -= 1; changed = True
                    reasons.append("Burn: Increasing Prod to balance clean Env.")
                else: break
            elif V[5] > 20:
                if int(V[0]) + dist[0] < 25:
                    dist[0] += 1; avail -= 1; changed = True
                    reasons.append("Burn: Cleaning Env.")
            else:
                if int(V[1]) + dist[1] > 5:
                    dist[1] -= 1; avail -= 1; changed = True
                    reasons.append("Burn: Sabotaging Prod (Safe sink).")
                else: break
            if avail + int(V[9]) <= 28: break

        # Transform to GUI Index Space (0-56)
        # Prod (index 1) needs +28, Special (index 5) needs +5
        final = [int(dist[0]), int(dist[1]) + 28, int(dist[2]), int(dist[3]), int(dist[4]), 5]
        if V[6] > 32: final[5] = 1; reasons.append("Pop: Control birth.") # 5 - 4 = 1
        elif V[6] < 18: final[5] = 10; reasons.append("Pop: Stimulate growth.") # 5 + 5 = 10
        
        return final, " | ".join(reasons)

    def predict_future(self, start_action, steps=5):
        import copy
        temp_env = copy.deepcopy(self.env)
        trajectory = []
        for _ in range(steps):
            avail = temp_env.unwrapped.V[temp_env.unwrapped.POINTS]
            act, _ = self.get_final_action(avail)
            action_for_env = list(act)
            action_for_env[temp_env.unwrapped.PRODUCTION] -= temp_env.unwrapped.Amin[temp_env.unwrapped.PRODUCTION]
            action_for_env[5] -= temp_env.unwrapped.Amin[5]
            
            # CRITICAL FIX: Convert to np.int64 array to satisfy gymnasium MultiDiscrete space
            action_array = np.array(action_for_env, dtype=np.int64)
            
            obs, _, term, trunc, _ = temp_env.unwrapped.step(action_array)
            V_next = obs + temp_env.unwrapped.Vmin
            trajectory.append(V_next)
            if term or trunc: break
        return trajectory

color_red = (220, 60, 60)
color_yellow = (230, 190, 50)  # Golden Yellow
color_blue = (50, 120, 240)
color_white = (15, 20, 28)     # Deep Dark Background
color_black = (240, 245, 255)  # Crystal White Text
color_turky = (30, 60, 80)     # Muted Deep Teal
color_light_blue = (60, 180, 240)
color_green = (40, 180, 100)
color_brown = (120, 100, 80)
color_orange = (220, 140, 40)


# --- need to be outside main for real randomness ---
# random_number = random.randint(0, 2)
# user_id = uuid.uuid4()


# def setup_game(number, id_):
#     u_id = get_number_value_by_name("bin/id.txt", "user_id")
#     if u_id == -1:
#         # hide data folder
#         os.system('attrib +h bin')
#
#         with open("bin/id.txt", "w") as file:
#             file.write(f"user_id: {id_}")
#
#         with open("bin/played_games.txt", "w") as file:
#             file.write(f"games_played: 0")
#
#         with open("bin/game_history.txt", "w") as file:
#             file.write("")
#
#         if number == 0:
#             with open("bin/config.txt", "w") as config:
#                 config.writelines("use_preview: 0\nuse_predict: 0\nuse_diagrams: 1\ngroup: 1")
#         elif number == 1:
#             with open("bin/config.txt", "w") as config:
#                 config.writelines("use_preview: 1\nuse_predict: 0\nuse_diagrams: 1\ngroup: 2")
#         elif number == 2:
#             with open("bin/config.txt", "w") as config:
#                 config.writelines("use_preview: 1\nuse_predict: 1\nuse_diagrams: 1\ngroup: 3")


def adapt_color(min_value, max_value, current):
    # red 255 0 0   -> 1
    #    255 128 0  -> 0.75
    # yellow 255 255 0 -> 0.5
    #       128 255 0  -> 0.25
    # green 0 255 0  -> 0
    if current > max_value / 2:
        factor = (current - (max_value / 2)) / (max_value / 2)
    else:
        factor = 1 - ((current - min_value) / ((max_value / 2) - min_value))
    return color_factor(factor)


def color_factor(factor):
    if factor > 0.5:
        red = 255
        green = 510 - (510 * factor)
    else:
        red = 255 * factor * 2
        green = 255
    return red, green, 0


def get_number_value_by_name(path, name):
    try:
        with open(path, "r") as config:
            number = ""
            size = len(name)
            line = config.readline(size)
            while line != "":
                if line == name:
                    text = config.readline()
                    for char in text:
                        if char.isdecimal():
                            number += char
                line = config.readline(size)
            if number == "":
                number = "-1"
            return int(number)
    except FileNotFoundError:
        return -1


def get_text_from_file(path):
    text = ""
    with open(path, "r") as file:
        current_line = file.readline()
        while current_line != "":
            text += f"{current_line}"
            current_line = file.readline()
    return text


def draw_text(pos, image, font_size, text, color=color_black):
    lines = text.splitlines()
    font = pygame.font.SysFont('Arial', font_size)
    for index, line in enumerate(lines):
        image.blit(font.render(line, True, color), (pos.x, pos.y + (font_size * index)))


def distribute1(action, points):
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


def transf_act_box(env, act):
    action_min = np.float32(np.array([0, -1, 0, 0, 0, -1]))
    action_max = np.float32(np.array([1, 1, 1, 1, 1, 1]))
    act = np.clip(act, action_min, action_max, dtype=np.float32)

    if act[1] < 0:
        act[1] = -act[1]
        reduce_production = True
    else:
        reduce_production = False

    regions_act = act[0:5]
    special_act = round(act[5] * 5)
    regions_act = distribute1(regions_act, env.unwrapped.V[env.unwrapped.POINTS])
    if reduce_production:
        regions_act[1] = -regions_act[1]

    for i in range(len(regions_act)):
        region_result = env.unwrapped.V[i] + regions_act[i]
        if region_result < env.unwrapped.Vmin[i]:
            regions_act[i] = env.unwrapped.Vmin[i] - env.unwrapped.V[i]
        elif region_result > env.unwrapped.Vmax[i]:
            regions_act[i] = env.unwrapped.Vmax[i] - env.unwrapped.V[i]

    act = np.append(regions_act, special_act)

    return act


class Camera(pygame.sprite.Group):
    def __init__(self):
        super().__init__()
        self.display_surface = pygame.display.get_surface()

        self.offset = Vector2()

        # these are the background image behind the play surface (toggle between them with key 1 = pygame.K_1):
        bg_files = ["gameboard.jpg", "gameboard_sepia.jpg", "gameboard_sepia2.jpg"]
        self.background = [None for k in range(len(bg_files))]
        for k, file in enumerate(bg_files):
            bg_path = os.path.join(ROOT_DIR, "assets", file)
            self.background[k] = pygame.image.load(bg_path).convert_alpha()
            self.background[k] = pygame.transform.scale(self.background[k], self.display_surface.get_size())
        self.background_rect = self.background[0].get_rect(topleft=(0, 0))

        # this is the background image when 'TAB' is hit (empty play surface, no sprites):
        empty_bg_path = os.path.join(ROOT_DIR, "assets", "empty_gameboard.jpg")
        self.empty_background = pygame.image.load(empty_bg_path).convert_alpha()
        self.empty_background = pygame.transform.scale(self.empty_background, self.display_surface.get_size())

        self.camera_movement_speed = 10
        self.zoom_factor = 1.0
        self.min_zoom = 1.0
        self.max_zoom = 3.0
        self.zoom_change = 0.1
        self.internal_surface = pygame.Surface(self.display_surface.get_size(), pygame.SRCALPHA)
        self.can_use_buttons = True
        self.show_background = True
        self.bg_selector = 0

    def custom_draw(self):
        self.internal_surface.fill(color_brown)

        offset = self.background_rect.topleft + self.offset
        if self.show_background:
            self.internal_surface.blit(self.background[self.bg_selector], offset)

            for sprite in self.sprites():
                offset_pos = sprite.rect.topleft + self.offset
                self.internal_surface.blit(sprite.image, offset_pos)
        else:
            self.internal_surface.blit(self.empty_background, offset)

        scaled_surface = pygame.transform.scale(self.internal_surface,
                                                Vector2(self.display_surface.get_size()) * self.zoom_factor)
        scaled_rect = scaled_surface.get_rect(
            center=(self.display_surface.get_width() // 2, self.display_surface.get_height() // 2))
        self.display_surface.blit(scaled_surface, scaled_rect)
        self.can_use_buttons = self.zoom_factor == self.min_zoom and self.show_background

    def keyboard_movement(self):
        keys = pygame.key.get_pressed()
        offset_range_x = pygame.display.get_window_size()[0] / 1920 * 640
        offset_range_y = pygame.display.get_window_size()[1] / 1080 * 360
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.offset.y += self.camera_movement_speed
            if self.offset.y > offset_range_y:
                self.offset.y = offset_range_y
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.offset.x += self.camera_movement_speed
            if self.offset.x > offset_range_x:
                self.offset.x = offset_range_x
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.offset.y -= self.camera_movement_speed
            if self.offset.y < -offset_range_y:
                self.offset.y = -offset_range_y
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.offset.x -= self.camera_movement_speed
            if self.offset.x < -offset_range_x:
                self.offset.x = -offset_range_x

    def mouse_zoom(self, scroll):
        change = -1
        if scroll > 0:
            change = 1
        zoom_change = change * self.zoom_change
        if self.min_zoom >= self.zoom_factor + zoom_change:
            self.zoom_factor = self.min_zoom
            self.offset = Vector2()
            self.show_background = True
        elif self.zoom_factor + zoom_change > self.max_zoom:
            self.zoom_factor = self.max_zoom
        else:
            self.zoom_factor += zoom_change


class Game:
    def __init__(self, camera, args):
        htl = dict_help_screens(args.language)
        self.dtl = dtl = dict_translate(args.language)
        self.camera = camera
        
        # Set Custom Icon
        try:
            icon_path = os.path.join(ROOT_DIR, "assets", "icon.png")
            icon = pygame.image.load(icon_path)
            pygame.display.set_icon(icon)
        except Exception as e:
            logger.warning(f"Could not load custom icon: {e}")
            
        self.env = gym.make('Oekolopoly-v2')
        self.agent_obs, _ = self.env.reset()
        
        # Lazy Load Torch to prevent hangs
        torch, RecurrentPPO = lazy_load_torch()
        
        model_path = os.path.join(ROOT_DIR, "sota_recurrent_champion.zip")
        if not os.path.exists(model_path):
            logger.error(f"CRITICAL: Model file {model_path} not found!")
            sys.exit(1)
        self.agent = RecurrentPPO.load(model_path, device='cpu')
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.current_action = [0, 0, 0, 0, 0, 0]
        self.all_actions = []
        self.action_font = pygame.font.SysFont('Times New Roman', 30)
        self.game_loop = True
        self.done = False
        self.available_actionpoints = self.env.unwrapped.V[self.env.unwrapped.POINTS]
        self.special_action = False
        self.special_action_points = 5
        self.preview_mode = False
        self.max_predict_usages = get_number_value_by_name("bin/config.txt", "predict_usages")
        self.predict_usages = self.max_predict_usages
        self.predict_used = False
        self.games_to_unlock_feature = 3

        # load config for features
        path = "bin/config.txt"
        diagrams_unlocked = get_number_value_by_name(path, "use_diagrams")
        preview_unlocked = get_number_value_by_name(path, "use_preview")
        predict_unlocked = get_number_value_by_name(path, "use_predict")
        group = get_number_value_by_name(path, "group")
        with open("bin/id.txt", "r") as file:
            usr_id = file.readline()
        self.config = f"{usr_id}, Group: {group}"

        if get_number_value_by_name("bin/played_games.txt", "games_played") < self.games_to_unlock_feature:
            self.diagrams_unlocked = 0
            self.preview_unlocked = 0
            self.predict_unlocked = 0
        else:
            self.diagrams_unlocked = diagrams_unlocked
            self.preview_unlocked = preview_unlocked
            self.predict_unlocked = predict_unlocked

        # buttons
        self.clear_action_button = Button(Vector2(1180, 470), Vector2(230, 40), color_orange, camera,
                                          dtl["ClearActions"], 20)
        self.step_button = Button(Vector2(600, 470), Vector2(190, 40), color_orange, camera, dtl["ExecuteStep"])
        self.close_game = Button(Vector2(1670, 15), Vector2(245, 55), color_red, camera, dtl["CloseGame"], 40)
        self.reset_button = Button(Vector2(400, 350), Vector2(170, 40), color_red, camera, dtl["Reset"])
        self.predict_button = Button(Vector2(600, 410), Vector2(190, 40), color_turky, camera,
                                     f"{dtl["BestMoveAI"]} ({self.predict_usages})")
        if self.max_predict_usages > 5:
            self.predict_button.text = dtl["BestMoveAI"]
        self.preview_button = Button(Vector2(600, 350), Vector2(210, 40), color_green, camera, dtl["PreviewMode"])
        self.game_instructions_button = Button(Vector2(20, 90), Vector2(240, 40), color_yellow, camera,
                                               dtl["GameInstructions"])
        self.sovereign_button = Button(Vector2(600, 290), Vector2(190, 40), color_turky, camera, "Sovereign AI", 25)
        # self.feedback_button = Button(Vector2(20, 145), Vector2(145, 40), color_yellow, camera, "Feedback?")
        self.game_history_button = Button(Vector2(20, 145), Vector2(190, 40), color_yellow, camera, dtl["GameHistory"])
        self.help_button = Button(Vector2(20, 200), Vector2(160, 40), color_yellow, camera, dtl["Help"])  # , 40)

        # diagrams
        sanitation_d = Diagram(Vector2(940, 30), camera, self.env.unwrapped.Vmin[0], self.env.unwrapped.Vmax[0])
        production_d = Diagram(Vector2(1390, 30), camera, self.env.unwrapped.Vmin[1], self.env.unwrapped.Vmax[1])
        education_d = Diagram(Vector2(1745, 580), camera, self.env.unwrapped.Vmin[2], self.env.unwrapped.Vmax[2])
        quality_of_life_d = Diagram(Vector2(980, 730), camera, self.env.unwrapped.Vmin[3], self.env.unwrapped.Vmax[3])
        population_growth_d = Diagram(Vector2(570, 730), camera, self.env.unwrapped.Vmin[4], self.env.unwrapped.Vmax[4])
        environment_d = Diagram(Vector2(1680, 130), camera, self.env.unwrapped.Vmin[5], self.env.unwrapped.Vmax[5])
        population_d = Diagram(Vector2(170, 720), camera, self.env.unwrapped.Vmin[6], self.env.unwrapped.Vmax[6])
        politics_d = Diagram(Vector2(100, 320), camera, self.env.unwrapped.Vmin[7], self.env.unwrapped.Vmax[7])
        action_points_d = Diagram(Vector2(330, 70), camera, self.env.unwrapped.Vmin[9], self.env.unwrapped.Vmax[9])
        self.diagrams = (
            sanitation_d, production_d, education_d, quality_of_life_d, population_growth_d, environment_d,
            population_d,
            politics_d, action_points_d)

        # action inputs
        sanitation_input = ActionInput(Vector2(880, 90), camera)
        production_input = ActionInput(Vector2(1330, 90), camera)
        education_input = ActionInput(Vector2(1685, 640), camera)
        quality_of_life_input = ActionInput(Vector2(920, 790), camera)
        population_growth_input = ActionInput(Vector2(510, 790), camera)
        self.action_inputs = (
            sanitation_input, production_input, education_input, quality_of_life_input, population_growth_input)

        # labels
        self.locked_preview_label = Label(Vector2(600, 350), Vector2(210, 40), camera, "        ?", 40, color_turky)
        self.locked_predict_label = Label(Vector2(600, 410), Vector2(190, 40), camera, "        ?", 40, color_turky)
        self.locked_preview_label.visible = False
        self.locked_predict_label.visible = False

        self.available_actionpoints_label = Label(Vector2(810, 410), Vector2(320, 40), camera, dtl["ActionPointsLeft"],
                                                  30, color_green, self.available_actionpoints)
        self.current_action_label = Label(Vector2(810, 470), Vector2(360, 40), camera, dtl["DistributedPoints"], 22,
                                          color_green)
        self.console_label = Label(Vector2(600, 530), Vector2(850, 30), camera, "", 20, color_yellow)
        self.preview_console_label = Label(Vector2(600, 580), Vector2(850, 30), camera, "", 20, color_yellow)
        self.round_label = Label(Vector2(420, 470), Vector2(140, 40), camera, dtl["Round"], 30, color_green)
        self.balance_label = Label(Vector2(1470, 470), Vector2(190, 40), camera, dtl["Balance"], 30, color_green)
        self.game_over_label = Label(Vector2(860, 580), Vector2(250, 70), camera, dtl["GameOver"], 50, color_red)

        # region labels with more info labels #
        action_points_label = Label(Vector2(390, 48), Vector2(155, 40), camera, dtl["APoints"], 23, color_white)
        more_info_action_points = MoreInfo(Vector2(390, 90), Vector2(170, 50),
                                           dtl["MoreInfoPoints"], camera)

        sanitation_label = Label(Vector2(815, 40), Vector2(115, 40), camera, dtl["Redevelop"], 23, color_white)
        #                                                                        "Sanierung"
        more_info_sanitation = MoreInfo(Vector2(750, 82), Vector2(180, 105),
                                        dtl["MoreInfoRedevelop"],
                                        camera)

        production_label = Label(Vector2(1250, 30), Vector2(122, 40), camera, dtl["Production"], 23, color_white)
        more_info_production = MoreInfo(Vector2(1227, 72), Vector2(145, 70),
                                        dtl["MoreInfoProduction"], camera)

        environment_label = Label(Vector2(1725, 380), Vector2(180, 40), camera, dtl["EnvirDamage"], 23, color_white)
        #                                                                           "Umweltbelastung"
        more_info_environment = MoreInfo(Vector2(1685, 422), Vector2(220, 105),
                                         dtl["MoreInfoEnvirDamage"], camera)

        education_label = Label(Vector2(1790, 830), Vector2(125, 40), camera, dtl["Enlightenment"], 23, color_white)
        #                                                                         "Aufklärung"
        more_info_education = MoreInfo(Vector2(1675, 872), Vector2(240, 105),
                                       dtl["MoreInfoEnlighten"], camera)

        quality_of_life_label = Label(Vector2(1025, 980), Vector2(155, 40), camera, dtl["QualityOfLife"], 23, color_white)
        more_info_quality_of_life = MoreInfo(Vector2(1025, 908), Vector2(210, 70),
                                             dtl["MoreInfoQuality"], camera)

        population_growth_label = Label(Vector2(615, 980), Vector2(185, 40), camera, dtl["ReproRate"], 23, color_white)
        more_info_population_growth = MoreInfo(Vector2(615, 928), Vector2(210, 50),
                                               dtl["MoreInfoReproRate"], camera)

        population_label = Label(Vector2(215, 980), Vector2(140, 40), camera, dtl["Population"], 23, color_white)
        more_info_population = MoreInfo(Vector2(215, 908), Vector2(240, 70),
                                        dtl["MoreInfoPopulation"], camera)

        politic_label = Label(Vector2(15, 435), Vector2(84, 40), camera, dtl["Politics"], 23, color_white)
        more_info_politic = MoreInfo(Vector2(15, 477), Vector2(90, 160),
                                     dtl["MoreInfoPolitics"], camera)
        # curr_action_label: Vector2(810, 470), Vector2(360, 40)
        more_info_curr_act = MoreInfo(Vector2(810, 520), Vector2(360, 80),
                                     dtl["MoreInfoCurrAction"], camera)


        self.region_labels = (
            action_points_label, sanitation_label, production_label, environment_label, education_label,
            quality_of_life_label, population_growth_label, population_label, politic_label, self.current_action_label)
            
        # MCTS Engine Initialization (Robust)
        try:
            self.torch_lib, self.ppo_class = lazy_load_torch()
            model_path = os.path.join(ROOT_DIR, "src", "sota_recurrent_champion.zip")
            if not os.path.exists(model_path):
                # Fallback to current dir
                model_path = os.path.join(ROOT_DIR, "sota_recurrent_champion.zip")
            
            logger.info(f"Loading AI Model from: {model_path}")
            self.ppo_model = self.ppo_class.load(model_path, device='cpu')
            self.mcts_engine = SovereignMCTS(self.ppo_model, num_simulations=100)
            self.guardian = SovereignGuardian(self.env)
            logger.info("Sovereign AI Engines initialized successfully.")
        except Exception as e:
            logger.error(f"AI Initialization Failed: {e}")
            self.mcts_engine = None
            self.guardian = None
            self.sovereign_button.color = color_red # Visually indicate AI is offline
        
        self.more_info_labels = (
            more_info_action_points, more_info_sanitation, more_info_production, more_info_environment,
            more_info_education, more_info_quality_of_life, more_info_population_growth,
            more_info_population, more_info_politic, more_info_curr_act)

        # special case labels and buttons
        self.special_population_growth_label = Label(Vector2(510, 640), Vector2(165, 35), camera, dtl["SpecialCase"], 30,
                                                     color_yellow)
        self.special_population_growth_decrease_button = Button(Vector2(510, 690), Vector2(35, 35), color_red, camera,
                                                                "-", 35)
        self.act_special_population_growth_label = Label(Vector2(570, 690), Vector2(35, 35), camera, "", 25,
                                                         color_yellow, "0")
        self.special_population_growth_increase_button = Button(Vector2(630, 690), Vector2(35, 35), color_green, camera,
                                                                "+", 35)
        # help screen
        self.help_screen = HelpScreen(camera, dtl, htl)
        self.help_screen.active = get_number_value_by_name("bin/played_games.txt", "games_played") == 0
        self.help_screen.active = True

        self.toggle_game_features()
        self.sovereign_reasoning_label = Label(Vector2(600, 630), Vector2(1000, 30), camera, "Sovereign AI Reasoning will appear here...", 18, color_light_blue)
        self.prediction_label = Label(Vector2(600, 670), Vector2(1000, 30), camera, "", 16, color_yellow)

    def check_button_press(self):
        if not self.help_screen.active:
            if not self.done:
                for action_input_index in range(len(self.action_inputs)):
                    self.change_current_action(action_input_index,
                                               self.action_inputs[action_input_index].button_pressed())

                if self.step_button.button_pressed():
                    self.run_simulation()
                if self.clear_action_button.button_pressed():
                    self.reset_current_action()
                if self.preview_button.button_pressed():
                    self.preview_mode = not self.preview_mode
                if self.special_action:
                    self.use_special_action_button()

                if self.sovereign_button.button_pressed():
                    self.sovereign_move()

                if self.max_predict_usages > 5:
                    # the use of button 'AI move' is allowed as often as desired:
                    if self.predict_button.button_pressed():
                        self.predict_next_move()
                else:
                    if self.predict_button.button_pressed():
                        # self.predict_usages is initialized above with self.max_predict_usages and decremented here
                        # for every valid predict_next_move. If self.predict_usages==0, then predict_next_move is
                        # inhibited.
                        if not self.predict_used and self.predict_usages > 0:
                            self.predict_usages -= 1
                            self.predict_used = True
                            self.predict_button.text = f"{self.dtl['BestMoveAI']} ({self.predict_usages})"
                        if self.predict_used:
                            self.predict_next_move()
                    if self.predict_used:
                        self.predict_button.color = color_light_blue
                    else:
                        self.predict_button.color = color_turky

            # always working unless in help mode
            if self.reset_button.button_pressed():
                self.reset_game()
            if self.help_button.button_pressed():
                self.help_screen.active = True

        # always working
        if self.close_game.button_pressed():
            self.game_loop = False
        # if self.feedback_button.button_pressed():
        #    os.startfile("feedback.txt", cwd="bin")
        if self.game_instructions_button.button_pressed():
            os.startfile("Spielanleitung.pdf")
        if self.game_history_button.button_pressed():
            os.startfile("current_game_history.txt", cwd="bin")

    def change_current_action(self, action_index, change):
        if change > 0 and self.current_action[action_index] + 1 + self.env.unwrapped.V[action_index] <= self.env.unwrapped.Vmax[action_index]:
            if self.current_action[action_index] < 0:
                self.available_actionpoints += 1
                self.current_action[action_index] += 1
            elif self.available_actionpoints > 0:
                self.current_action[action_index] += 1
                self.available_actionpoints -= 1
        elif change < 0 and self.current_action[action_index] - 1 + self.env.unwrapped.V[action_index] >= self.env.unwrapped.Vmin[action_index]:
            if self.current_action[action_index] > 0:
                self.available_actionpoints += 1
                self.current_action[action_index] -= 1
            elif self.available_actionpoints > 0 and action_index == 1:
                self.current_action[action_index] -= 1
                self.available_actionpoints -= 1

    def checkHoverOnLabels(self):
        for label_index in range(len(self.region_labels)):
            label_rect = pygame.Rect(self.region_labels[label_index])
            label_rect.topleft += self.camera.offset
            if label_rect.collidepoint(pygame.mouse.get_pos()):
                self.more_info_labels[label_index].active = True
            else:
                self.more_info_labels[label_index].active = False

    def toggle_special_labels(self):
        self.special_population_growth_label.visible = self.special_action
        self.act_special_population_growth_label.visible = self.special_action
        self.special_population_growth_increase_button.visible = self.special_action
        self.special_population_growth_decrease_button.visible = self.special_action

    def use_special_action_button(self):
        if self.special_population_growth_increase_button.button_pressed():
            if self.current_action[5] < 0:
                self.special_action_points += 1
                self.current_action[5] += 1
            elif self.special_action_points > 0:
                self.current_action[5] += 1
                self.special_action_points -= 1
        elif self.special_population_growth_decrease_button.button_pressed():
            if self.current_action[5] > 0:
                self.special_action_points += 1
                self.current_action[5] -= 1
            elif self.special_action_points > 0:
                self.special_action_points += 1
                self.current_action[5] -= 1

    def update_diagrams(self):
        for diagram_index in range(len(self.diagrams)):
            env_position = diagram_index
            if env_position == 8:
                env_position += 1
            self.diagrams[diagram_index].current_value = self.env.unwrapped.V[env_position]

    def update_labels(self):
        self.round_label.variable_text = self.env.unwrapped.V[self.env.unwrapped.ROUND]
        self.available_actionpoints_label.variable_text = self.available_actionpoints
        if self.available_actionpoints == 0:
            self.available_actionpoints_label.background_color = color_red
        else:
            self.available_actionpoints_label.background_color = color_green
        self.toggle_special_labels()
        self.preview_console_label.visible = self.preview_mode

        if self.done:
            self.game_over_label.visible = True
            self.preview_console_label.visible = False
            self.reset_button.color = color_green
        else:
            self.game_over_label.visible = False
            self.reset_button.color = color_red

        distr_points = self.current_action.copy()       # /WK/2025-12-12, might later change display values
        if self.special_action:
            self.current_action_label.variable_text = distr_points
            self.act_special_population_growth_label.variable_text = self.current_action[5]
        else:
            self.current_action_label.variable_text = distr_points # [0:5]   /WK/2025-12-12

        for label_index in range(len(self.action_inputs)):
            number_change = self.current_action[label_index]
            if number_change > 0:
                text = f"+{number_change}"
            else:
                text = number_change
            self.action_inputs[label_index].current_label.variable_text = text

        self.update_preview_labels()

    def update(self) -> None:
        if self.camera.can_use_buttons:
            self.check_button_press()
            if not self.help_screen.active:
                self.checkHoverOnLabels()
        self.update_diagrams()
        self.update_labels()
        self.check_new_help()
        if self.preview_mode:
            self.preview_action()

    def predict_next_move(self) -> None:
        obs_for_agent = self.agent_obs + self.env.unwrapped.Vmin
        # Handle recurrent predict (expects batch dimension)
        obs_batch = np.array([obs_for_agent])
        agent_action, self.lstm_states = self.agent.predict(
            obs_batch, 
            state=self.lstm_states, 
            episode_start=self.episode_starts, 
            deterministic=True
        )
        self.episode_starts = np.zeros((1,), dtype=bool)
        
        a_for_env = transf_act_box(self.env, agent_action[0])
        a_for_env = [int(a_for_env[i]) for i in range(a_for_env.size)]
        self.current_action = a_for_env
        for i in range(len(a_for_env)-1):
            self.available_actionpoints -= abs(a_for_env[i])
        self.special_action_points = 5 - abs(self.current_action[5])

    def sovereign_move(self) -> None:
        """Executes a high-precision MCTS search to find the optimal move."""
        if self.mcts_engine is None:
            msg = "Sovereign AI is OFFLINE (Init Failed). Check logs."
            logger.error(msg)
            self.sovereign_reasoning_label.variable_text = msg
            return

        logger.info("Sovereign AI is calculating the optimal trajectory...")
        
        # 1. Search for the best raw action using MCTS
        best_raw_action = self.mcts_engine.search(self.env)
        
        # 2. Map to final valid action using Guardian (returns Indices)
        avail = int(self.env.unwrapped.V[self.env.unwrapped.POINTS])
        a_for_env, reason = self.guardian.get_final_action(best_raw_action, avail)
        
        # 3. Update GUI state
        self.current_action = a_for_env
        self.available_actionpoints = self.env.unwrapped.V[self.env.unwrapped.POINTS]
        for i in range(5):
            self.available_actionpoints -= abs(a_for_env[i] - (28 if i==1 else 0))
        self.special_action_points = 5 - abs(self.current_action[5] - 5)
        
        self.sovereign_reasoning_label.variable_text = f" MCTS Selection: {reason}"
        
        # Predictive Analytics
        trajectory = self.guardian.predict_future(a_for_env, steps=5)
        pred_text = f" {self.dtl['Prediction']}: "
        for v in trajectory:
            pred_text += f"Y{int(v[8])}:[E:{int(v[5])},Q:{int(v[3])}] "
        self.prediction_label.variable_text = pred_text
        
        self.run_simulation()

    def preview_action(self):
        temp_env = copy.deepcopy(self.env)
        action = list(self.current_action)

        action[temp_env.unwrapped.PRODUCTION] -= temp_env.unwrapped.Amin[temp_env.unwrapped.PRODUCTION]
        action[5] -= temp_env.unwrapped.Amin[5]
        import numpy as np
        action_array = np.array(action, dtype=np.int64)
        obs, rew, terminated, truncated, info = temp_env.unwrapped.step_w_o_clip(action_array)
        temp_V = obs + temp_env.unwrapped.Vmin
        done = terminated or truncated
        for diagram_index in range(len(self.diagrams)):
            env_index = diagram_index
            if env_index == 8:
                env_index += 1
            self.diagrams[diagram_index].done = done
            if done:
                self.preview_console_label.variable_text = (f"{self.dtl["AttentionGameOver"]}: "
                                                            f"{info['done_reason']}{info['done_reason_detail']}")
            else:
                self.preview_console_label.variable_text = ""
            # self.diagrams[diagram_index].next_value = temp_env.unwrapped.V[env_index]   # bug before 2025-12-13.
            #
            # Bug fix: set the next value occording to temp_V, the resulting V after step_w_o_clip. Only then, an
            # attempt to set special case 's=+4' in case 'Education=20', which allows only 's=+-3' will have the same
            # resulting Popul_Growth as 's=+3' as it should:
            self.diagrams[diagram_index].next_value = temp_V[env_index]  # the unclipped values
            # --- only for debug: ---
            # if env_index == 9:
            #     next_round = temp_env.unwrapped.V[8]
            #     print(f"Next round {next_round} action points: {self.diagrams[diagram_index].next_value}")

    def update_preview_labels(self):
        if self.preview_mode:
            self.preview_button.color = color_green
        else:
            self.preview_button.color = color_red
        for diagram_index in range(len(self.diagrams)):
            env_index = diagram_index
            if env_index == 8:
                env_index += 1
            self.diagrams[diagram_index].preview_mode = self.preview_mode

    def toggle_game_features(self):
        if get_number_value_by_name("bin/played_games.txt", "games_played") >= self.games_to_unlock_feature:
            path = "bin/config.txt"
            self.diagrams_unlocked = get_number_value_by_name(path, "use_diagrams")
            self.preview_unlocked = get_number_value_by_name(path, "use_preview")
            self.predict_unlocked = get_number_value_by_name(path, "use_predict")
            self.locked_predict_label.visible = False
            self.locked_preview_label.visible = False
        else:
            if get_number_value_by_name("bin/config.txt", "use_preview"):
                self.locked_preview_label.visible = True
            if get_number_value_by_name("bin/config.txt", "use_predict"):
                self.locked_predict_label.visible = True

        self.predict_button.visible = self.predict_unlocked
        self.preview_button.visible = self.preview_unlocked
        for diagram in self.diagrams:
            diagram.diagram_unlocked = self.diagrams_unlocked

        if get_number_value_by_name("bin/played_games.txt", "games_played") == 3:
            self.help_screen.active = True
            self.help_screen.help_step = 8

    def check_new_help(self):
        if get_number_value_by_name("bin/played_games.txt", "games_played") >= self.games_to_unlock_feature:
            if get_number_value_by_name("bin/config.txt", "use_predict"):
                self.help_screen.max_help_steps = 10
            elif get_number_value_by_name("bin/config.txt", "use_preview"):
                self.help_screen.max_help_steps = 9
            else:
                self.help_screen.max_help_steps = 8
        else:
            self.help_screen.max_help_steps = 7

    def run_simulation(self):
        """Standard human step execution with logging."""
        # Actions are already indices (0-56), just pass them to env.step
        action_array = np.array(self.current_action, dtype=np.int64)
        
        try:
            logger.info(f"Executing Move: {self.current_action}")
            self.agent_obs, reward, self.done, truncated, info = self.env.step(action_array)
            self.all_actions.append(self.current_action.copy())
            self.reset_current_action()
            if self.done:
                reason = info.get('done_reason', 'Unknown')
                detail = info.get('done_reason_detail', '')
                logger.info(f"Simulation Terminated. Reason: {reason} | Detail: {detail}")
                self.handle_game_over(info)
        except Exception as e:
            error_msg = f"Simulation Error: {e}"
            logger.error(error_msg)
            if hasattr(self, 'console_label'):
                self.console_label.variable_text = error_msg

    def handle_game_over(self, info: dict):
        """Processes the end of a game, calculates harmony, and saves a report."""
        V = self.env.unwrapped.V
        year = int(V[8])
        if year >= 30:
            grade, penalty = self.calculate_harmony_score()
            msg = f"VICTORY! {self.dtl['HarmonyGrade']}: {grade} ({self.dtl['Penalty']}: {penalty:.1f})"
            self.console_label.variable_text = msg
            logger.info(msg)
            self.save_victory_report(grade, penalty)
        else:
            self.console_label.variable_text = f"GAME OVER: {info.get('done_reason')}"

    def calculate_harmony_score(self) -> Tuple[str, float]:
        """Calculates the final harmony grade based on V106 standards."""
        V = self.env.unwrapped.V
        # Penalties relative to ideal targets
        q_p = abs(V[3] - 15)
        e_p = abs(V[5] - 15)
        ed_p = 29 - V[2]
        p_p = abs(V[6] - 25)
        total_penalty = q_p + e_p + ed_p + p_p
        
        grade = "D"
        if total_penalty < 5: grade = "A+"
        elif total_penalty < 10: grade = "A"
        elif total_penalty < 15: grade = "B"
        elif total_penalty < 20: grade = "C"
        return grade, float(total_penalty)

    def save_victory_report(self, grade: str, penalty: float):
        """Saves a detailed report of the successful run."""
        reports_dir = os.path.join(ROOT_DIR, "reports")
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        V = self.env.unwrapped.V
        report_path = os.path.join(reports_dir, f"victory_{int(V[8])}_years_{grade}.txt")
        with open(report_path, "w") as f:
            f.write("--- SOVEREIGN CHAMPION VICTORY REPORT ---\n")
            f.write(f"Grade: {grade}\nPenalty: {penalty:.2f}\n")
            f.write(f"Final States: {V.tolist()}\n")
            f.write(f"Total Steps: {len(self.all_actions)}\n")
        logger.info(f"Victory report saved to {report_path}")

    def reset_current_action(self):
        self.current_action = [0, 0, 0, 0, 0, 0]
        self.available_actionpoints = self.env.unwrapped.V[self.env.unwrapped.POINTS]
        self.special_action_points = 5

    def reset_game(self):
        # if not self.done and self.all_actions != []:
        # date_now = datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        # text = f"Vorzeitiger Abbruch des Spiels, Runde: {self.env.unwrapped.V[8]}, Zustaende {self.env.unwrapped.V}, Datum: {date_now}, Zuege: {self.all_actions}\n"
        # with open("bin/game_history.txt", "a") as history:
        #     history.write(text)

        self.env = gym.make('Oekolopoly-v2')
        self.agent_obs, _ = self.env.reset()
        self.all_actions = []
        self.current_action = [0, 0, 0, 0, 0, 0]
        self.console_label.text = ""
        self.available_actionpoints = self.env.unwrapped.V[self.env.unwrapped.POINTS]
        self.done = False
        self.special_action = False
        self.special_action_points = 5
        # self.preview_mode = False
        self.predict_usages = self.max_predict_usages
        self.predict_used = False
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.toggle_game_features()


class Button(pygame.sprite.Sprite):
    def __init__(self, pos, size, color, camera, text, font_size=30):
        super().__init__(camera)
        self.size = Vector2(size.x / 1920 * pygame.display.get_window_size()[0],
                            size.y / 1080 * pygame.display.get_window_size()[1])
        self.image = pygame.Surface(self.size)
        self.image.fill(color)
        self.color = color
        self.camera = camera
        self.rect = self.image.get_rect()
        self.rect.topleft = (
            pos.x / 1920 * pygame.display.get_window_size()[0], pos.y / 1080 * pygame.display.get_window_size()[1])
        self.is_pressed = False
        self.font_size = int(font_size / 1920 * pygame.display.get_window_size()[0])
        self.button_text_font = pygame.font.SysFont('Times New Roman', self.font_size)
        self.text = text
        self.visible = True

    def update(self, *args, **kwargs):
        if self.visible:
            self.draw()
        else:
            self.image = pygame.Surface(Vector2(0, 0))

    def draw(self):
        self.image = pygame.Surface(self.size)
        self.image.fill(self.color)
        text_position = (self.size.y - self.font_size - 5) / 2
        self.image.blit(self.button_text_font.render(f"{self.text}", True, (250, 250, 250)),
                        (10 / 1920 * pygame.display.get_window_size()[0], text_position))
        pygame.draw.rect(self.image, (240, 240, 240), pygame.Rect((0, 0), self.size), 1)

    def button_pressed(self) -> bool:
        if not self.visible:
            return False
        pressed = False
        button_rect = pygame.Rect(self.rect)
        button_rect.topleft += self.camera.offset
        if button_rect.collidepoint(pygame.mouse.get_pos()):
            if pygame.mouse.get_pressed()[0] == 1 and not self.is_pressed:
                self.is_pressed = True
                pressed = True
        if pygame.mouse.get_pressed()[0] == 0:
            self.is_pressed = False
        return pressed


class Label(pygame.sprite.Sprite):
    def __init__(self, pos, size, camera, text, font_size, background_color, variable_text=""):
        super().__init__(camera)
        self.size = Vector2(size.x / 1920 * pygame.display.get_window_size()[0],
                            size.y / 1080 * pygame.display.get_window_size()[1])
        self.image = pygame.Surface(self.size)
        self.image.fill(background_color)
        self.background_color = background_color
        self.text = text
        self.variable_text = variable_text
        self.rect = self.image.get_rect()
        self.rect.topleft = (
            pos.x / 1920 * pygame.display.get_window_size()[0], pos.y / 1080 * pygame.display.get_window_size()[1])
        self.font_size = int(font_size / 1920 * pygame.display.get_window_size()[0])
        self.text_font = pygame.font.SysFont('Times New Roman', self.font_size)
        self.visible = True

    def update(self, *args, **kwargs):
        if self.visible:
            self.draw()
        else:
            self.image = pygame.Surface(Vector2(0, 0))

    def draw(self):
        self.image = pygame.Surface(self.size)
        self.image.fill(self.background_color)
        text_position = (self.size.y - self.font_size - 5) / 2
        # Use white text for labels by default in dark mode
        self.image.blit(self.text_font.render(f"{self.text}{self.variable_text}", True, (240, 245, 255)),
                        (10 / 1920 * pygame.display.get_window_size()[0], text_position))
        pygame.draw.rect(self.image, color_yellow, pygame.Rect((0, 0), self.size), 1) # Gold border


class Diagram(pygame.sprite.Sprite):
    def __init__(self, pos, camera, min_value, max_value):
        super().__init__(camera)
        self.size = Vector2(37 / 1920 * pygame.display.get_window_size()[0],
                            300 / 1080 * pygame.display.get_window_size()[1])
        self.image = pygame.Surface(self.size)
        self.value_color = color_black
        self.image.fill(color_white)
        self.rect = self.image.get_rect()
        self.rect.topleft = (
            pos.x / 1920 * pygame.display.get_window_size()[0], pos.y / 1080 * pygame.display.get_window_size()[1])
        self.button_text_font = pygame.font.SysFont('Times New Roman',
                                                    int(20 / 1920 * pygame.display.get_window_size()[0]))
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        self.next_value = 0
        self.preview_mode = False
        self.done = False
        self.diagram_unlocked = 1

        self.current_value_label = Label(pos + Vector2(45, 123), Vector2(70, 60), camera, "", 40,
                                         color_green, str(self.current_value))
        self.preview_label = Label(pos + Vector2(45, 63), Vector2(70, 45), camera, "", 30,
                                   color_green)
        self.preview_label_scaled_pos = self.preview_label.rect.topleft

    def update(self, *args, **kwargs):
        if self.preview_mode:
            self.update_preview()
            self.preview_label.visible = True
        else:
            self.preview_label.visible = False
        self.draw()

    def update_preview(self):
        value_change = self.next_value - self.current_value
        if self.next_value < self.min_value or self.next_value > self.max_value:
            color = color_red
        elif self.done:
            color = color_orange
        else:
            color = color_green
        self.preview_label.rect.topleft = self.preview_label_scaled_pos
        if value_change < 0:
            value_change *= -1
            self.preview_label.variable_text = f"- {value_change}"
            self.preview_label.rect.topleft = self.preview_label_scaled_pos + Vector2(0, 135 / 1080 * pygame.display.get_window_size()[1])
        elif value_change > 0:
            self.preview_label.variable_text = f"+ {value_change}"
        else:
            self.preview_label.rect.topleft = Vector2(8000, 0)
        self.preview_label.background_color = color

    def draw(self):
        self.image.fill(color_white)

        self.current_value_label.variable_text = self.current_value

        if self.diagram_unlocked:
            diagram_height = self.size.y
            diagram_width = self.size.x
            boarder_size = 3

            # diagram border (Gold)
            pygame.draw.rect(self.image, color_yellow, pygame.Rect(0, 0, diagram_width, diagram_height), boarder_size)

            # update color
            self.value_color = adapt_color(self.min_value, self.max_value, self.current_value)
            self.current_value_label.background_color = adapt_color(self.min_value, self.max_value, self.current_value)

            if self.min_value < 0:
                current_value_diagram_size = diagram_height / (self.max_value - self.min_value + 1) * (
                        self.current_value - self.min_value)
            else:
                current_value_diagram_size = diagram_height / self.max_value * self.current_value
            # diagram bar
            pygame.draw.rect(self.image, self.value_color,
                             pygame.Rect(boarder_size, diagram_height - current_value_diagram_size + boarder_size - 1,
                                         diagram_width - 2 * boarder_size,
                                         current_value_diagram_size - 2 * (boarder_size - 1)))

            # preview bar
            if self.preview_mode:
                value_change = self.next_value - self.current_value
                if self.min_value < 0:
                    current_value_diagram_size = diagram_height / (self.max_value + - self.min_value) * (
                            self.next_value - self.min_value)
                else:
                    current_value_diagram_size = diagram_height / self.max_value * self.next_value
                if value_change != 0:
                    pygame.draw.rect(self.image, color_black,
                                     pygame.Rect(boarder_size, diagram_height - current_value_diagram_size + 3,
                                                 diagram_width - 2 * boarder_size, 3))

            # diagram max/min values
            self.image.blit(self.button_text_font.render(f"{self.max_value}", True, color_black), (5, 5))
            self.image.blit(self.button_text_font.render(f"{self.min_value}", True, color_black),
                            (5, self.rect.size[1] - (30 / 1920 * pygame.display.get_window_size()[0])))
        else:
            self.image.fill(color_turky)
            font = pygame.font.SysFont('Times New Roman', int(50 / 1920 * pygame.display.get_window_size()[0]))
            self.image.blit(font.render("?", True, color_black), (5, 115 / 1920 * pygame.display.get_window_size()[0]))
            pygame.draw.rect(self.image, color_black, pygame.Rect((0, 0), self.size), 2)


class ActionInput:
    def __init__(self, pos, camera):
        self.position = pos
        self.increase_button = Button(Vector2(pos.x, pos.y + 5), Vector2(50, 50), color_green, camera, "+", 45)
        self.decrease_button = Button(Vector2(pos.x, pos.y + 125), Vector2(50, 50), color_red, camera, "-", 45)
        self.current_label = Label(Vector2(pos.x, pos.y + 65), Vector2(50, 50), camera, "", 24, color_yellow, "0")

    def button_pressed(self):
        if self.increase_button.button_pressed():
            return 1
        elif self.decrease_button.button_pressed():
            return -1
        else:
            return 0


# TODO Geplant: Pfeile für Wirkungsgefüge (siehe Seite 2 in Spielanleitung)
class MoreInfo(pygame.sprite.Sprite):
    def __init__(self, pos, size, text, camera):
        """
        Window with more info that appears whenever ``self.active==True`` (usually when 'hover'-ing another element)

        :param pos: position (x,y) in (0..1920, 0..1080) of window
        :param size: (w,h) in pixels of window
        :param text: (multiline) text to display in window
        :param camera: needed to initialize superclass
        """
        super().__init__(camera)
        self.image = pygame.Surface(size)
        self.size = Vector2(size.x / 1920 * pygame.display.get_window_size()[0],
                            size.y / 1080 * pygame.display.get_window_size()[1])
        self.image.fill(color_white)
        self.rect = self.image.get_rect()
        self.rect.topleft = (
            pos.x / 1920 * pygame.display.get_window_size()[0], pos.y / 1080 * pygame.display.get_window_size()[1])
        self.text = text
        self.active = False
        self.scale_x = pygame.display.get_window_size()[0] / 1920
        self.scale_y = pygame.display.get_window_size()[1] / 1080

    def update(self, *args, **kwargs):
        if self.active:
            self.draw()
        else:
            self.image = pygame.Surface(Vector2(0, 0))

    def draw(self):
        self.image = pygame.Surface(self.size)
        self.image.fill(color_white)
        font_size = 18
        draw_text(Vector2(5, 5), self.image, int(font_size / 1920 * pygame.display.get_window_size()[0]), self.text,
                  color_black)
        pygame.draw.rect(self.image, color_black, pygame.Rect((0, 0), self.size), 2)


class HelpScreen(pygame.sprite.Sprite):
    def __init__(self, camera, dtl: dict, htl: dict):
        super().__init__(camera)
        self.dtl = dtl
        self.htl = htl
        self.image = pygame.Surface(pygame.display.get_window_size()).convert_alpha()
        self.image.fill((0, 0, 0, 0))
        self.rect = self.image.get_rect()
        self.rect.topleft = Vector2()
        self.active = False
        self.camera = camera
        self.scale_x = pygame.display.get_window_size()[0] / 1920
        self.scale_y = pygame.display.get_window_size()[1] / 1080
        self.exit_help_button = Button(Vector2(890, 850), Vector2(200, 40), color_red, camera, text=dtl["ExitHelp"])
        self.help_mode = Label(Vector2(860, 900), Vector2(260, 80), camera, dtl["HelpMode"], 50, color_yellow)
        self.step_back_button = Button(Vector2(760, 1000), Vector2(150, 40), color_green, camera, text=dtl["Back"])
        self.page_label = Label(Vector2(910, 1000), Vector2(160, 40), camera, dtl["Page"], 30, color_turky)
        self.step_forward_button = Button(Vector2(1070, 1000), Vector2(160, 40), color_green, camera, dtl["Forward"])

        # special case labels and buttons as preview if not unlocked yet
        self.special_label = Label(Vector2(510, 640), Vector2(165, 35), camera, dtl["SpecialCase"], 30, color_yellow)
        self.special_negative_label = Label(Vector2(510, 690), Vector2(35, 35), camera, "-", 35, color_red)
        self.special_number_label = Label(Vector2(570, 690), Vector2(35, 35), camera, "", 25, color_yellow, "0")
        self.special_positive_label = Label(Vector2(630, 690), Vector2(35, 35), camera, "+", 35, color_green)
        special_case_pic_path = os.path.join(ROOT_DIR, "assets", "education_population_growth.jpg")
        self.special_case_picture = pygame.image.load(special_case_pic_path).convert_alpha()
        self.special_case_picture = pygame.transform.scale(self.special_case_picture,
                                                           (400 * self.scale_x, 400 * self.scale_y))

        # preview labels
        self.up_preview_label = Label(Vector2(985, 90), Vector2(70, 45), camera, "+4", 30, color_green)
        self.down_preview_label = Label(Vector2(985, 90 + 135), Vector2(70, 45), camera, "-5", 30, color_orange)
        self.preview_console_label = Label(Vector2(600, 580), Vector2(850, 30), camera, "", 20, color_yellow)

        self.help_step = 1
        self.max_help_steps = 7

    def update(self, *args, **kwargs):
        self.toggle_buttons()
        self.toggle_help_preview_labels()
        if self.active:
            if self.camera.can_use_buttons:
                self.check_inputs()
            self.update_buttons_and_labels()
            self.draw()

    def check_inputs(self):
        if self.step_forward_button.button_pressed():
            self.help_step += 1
        elif self.step_back_button.button_pressed():
            if self.help_step > 1:
                self.help_step -= 1

        if self.exit_help_button.button_pressed() or self.help_step > self.max_help_steps:
            self.active = False
            self.help_step = 1

    def update_buttons_and_labels(self):
        self.page_label.variable_text = f"{self.help_step}/{self.max_help_steps}"
        if self.help_step == self.max_help_steps:
            self.step_forward_button.text = self.dtl["ExitHelp"]
            self.step_forward_button.color = color_red
        else:
            self.step_forward_button.text = self.dtl["Forward"]
            self.step_forward_button.color = color_green
        if self.help_step == 1:
            self.step_back_button.rect.topleft = Vector2(3000, 0)
        else:
            self.step_back_button.rect.topleft = Vector2(760 * self.scale_x, 1000 * self.scale_y)

    def toggle_buttons(self):
        if self.active:
            self.step_forward_button.visible = True
            self.exit_help_button.visible = True
            self.step_back_button.visible = True
            self.page_label.visible = True
            self.help_mode.visible = True
        else:
            self.step_forward_button.visible = False
            self.exit_help_button.visible = False
            self.step_back_button.visible = False
            self.page_label.visible = False
            self.help_mode.visible = False

    def toggle_help_preview_labels(self):
        self.special_label.visible = False
        self.special_number_label.visible = False
        self.special_negative_label.visible = False
        self.special_positive_label.visible = False
        self.up_preview_label.visible = False
        self.down_preview_label.visible = False
        self.preview_console_label.visible = False

        if self.active:
            if self.help_step == 5:
                self.special_label.visible = True
                self.special_number_label.visible = True
                self.special_negative_label.visible = True
                self.special_positive_label.visible = True
            elif self.help_step == 9:
                self.up_preview_label.visible = True
                self.down_preview_label.visible = True
                self.preview_console_label.visible = True

    def draw(self):
        self.image.fill((0, 0, 0, 0))
        x = self.scale_x
        y = self.scale_y
        small_font_size = int(25 * x)
        big_font_size = int(30 * x)
        # extra_large_font_size = int(50 * x)
        if self.active:
            # grey background
            pygame.draw.rect(self.image, (0, 0, 0, 200), pygame.Rect(0, 0, 1920 * x, 1080 * y))
            pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(1670 * x, 15 * y, 245 * x, 55 * y))
            # played_games = get_number_value_by_name("bin/played_games.txt", "games_played")

            if self.help_step == 1:
                draw_text(Vector2(400 * x, 200 * y), self.image, big_font_size, self.htl["hs1.0"])
            elif self.help_step == 2:
                #                                                        #      left,    top,   width,  height
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(805 * x, 15 * y, 280 * x, 330 * y))
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(805 * x, 405 * y, 610 * x, 110 * y))
                draw_text(Vector2(450 * x, 15 * y), self.image, small_font_size, self.htl["hs2.0"])
                draw_text(Vector2(1100 * x, 20 * y), self.image, small_font_size, self.htl["hs2.1"])
                draw_text(Vector2(1100 * x, 155 * y), self.image, small_font_size, self.htl["hs2.2"])
                draw_text(Vector2(805 * x, 520 * y), self.image, small_font_size, self.htl["hs2.3"])
            elif self.help_step == 3:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(415 * x, 465 * y, 150 * x, 50 * y))
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(395 * x, 345 * y, 180 * x, 50 * y))
                draw_text(Vector2(585 * x, 340 * y), self.image, small_font_size,
                          self.htl["hs3.0"])
            elif self.help_step == 4:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(595 * x, 465 * y, 580 * x, 50 * y))
                draw_text(Vector2(595 * x, 520 * y), self.image, small_font_size,
                          self.htl["hs4.0"])
            elif self.help_step == 5:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(505 * x, 635 * y, 175 * x, 95 * y))
                self.image.blit(self.special_case_picture, (790 * x, 200 * y))
                draw_text(Vector2(690 * x, 625 * y), self.image, small_font_size,
                          self.htl["hs5.0"])
            elif self.help_step == 6:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(595 * x, 525 * y, 860 * x, 40 * y))
                draw_text(Vector2(595 * x, 570 * y), self.image, small_font_size,
                          self.htl["hs6.0"])
            elif self.help_step == 7:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(15 * x, 85 * y, 250 * x, 162 * y))
                draw_text(Vector2(275 * x, 80 * y), self.image, small_font_size,
                          self.htl["hs7.0"])
            elif self.help_step == 8:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(805 * x, 15 * y, 280 * x, 330 * y))
                draw_text(Vector2(1100 * x, 70 * y), self.image, small_font_size,
                          self.htl["hs8.0"])
                # s = ""
                # s2 = "s"
                # if self.max_help_steps == 10:
                #    features = "Balken Feature, Vorschaumodus und Bester Zug."
                # elif self.max_help_steps == 9:
                #    features = "Balken Feature und Vorschaumodus."
                # else:
                #    features = "Balken Feature."
                #    s = "s"
                #    s2 = ""
                # if played_games == 3:
                #    draw_text(Vector2(500 * x, 500 * y), self.image, extra_large_font_size,
                #              f"Hurra du hast folgende{s} neue{s} Feature{s2} freigeschaltet!\n"
                #              f"{features}\n\n"
                #              f"Dazu gibt es neue Hilfeseiten!", color_green)
            elif self.help_step == 9:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(595 * x, 345 * y, 220 * x, 50 * y))
                draw_text(Vector2(370 * x, 280 * y), self.image, small_font_size, self.htl["hs9.0"])
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(805 * x, 15 * y, 280 * x, 330 * y))
                pygame.draw.rect(self.image, color_black, pygame.Rect(941 * x, 280 * y, 34 * x, 3 * y))
                draw_text(Vector2(1095 * x, 50 * y), self.image, small_font_size, self.htl["hs9.1"])
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(595 * x, 575 * y, 860 * x, 40 * y))
                draw_text(Vector2(595 * x, 620 * y), self.image, small_font_size, self.htl["hs9.2"])
            elif self.help_step == 10:
                text = (self.htl["hs10.0"])
                max_predict_uses = get_number_value_by_name("bin/config.txt", "predict")
                if max_predict_uses <= 5:
                    text += f"{self.htl['hs10.1']} {max_predict_uses}{self.htl['hs10.2']}"
                    # this prints (e.g.): "Attention: Use of AI is limited to 4 times per game!"
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(595 * x, 405 * y, 200 * x, 50 * y))
                draw_text(Vector2(595 * x, 460 * y), self.image, small_font_size, text)


def parse_args():
    """
    Parse arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Parameters for Oekolopoly GUI')
    parser.add_argument('--language', type=str, default="en",
                        help='Language of GUI')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # setup_game(random_number, user_id)

    fps = 60
    pygame.init()
    pygame.display.set_mode()
    # pygame.display.set_mode((1920, 1080))
    pygame.display.set_caption("Ökolopoly")
    clock = pygame.time.Clock()

    camera = Camera()
    game = Game(camera, args)

    game_loop = True
    while game_loop:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_loop = False

            if event.type == pygame.MOUSEWHEEL:
                camera.mouse_zoom(event.y)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    camera.show_background = not camera.show_background
                if event.key == pygame.K_1:
                    camera.bg_selector = (camera.bg_selector + 1) % len(camera.background)
                    # print(f"bg_selector = {camera.bg_selector}")

        # update game
        game.update()
        if not game.game_loop:
            game_loop = False

        # print(pygame.mouse.get_pos())

        camera.keyboard_movement()

        # draw
        camera.custom_draw()
        camera.update()

        # update window
        pygame.display.update()
        clock.tick(fps)

    # prepare and send email
    email = ""
    history_text = get_text_from_file("bin/game_history.txt")
    if len(history_text) > 0:
        email += history_text
        with open("bin/game_history.txt", "w") as file:
            file.write("")
    feedback_text = get_text_from_file("bin/feedback.txt")
    if len(feedback_text) > 0:
        email += f"\nFeedback\n {feedback_text}"
        with open("bin/feedback.txt", "w") as file:
            file.write("")

    # if len(email) > 12:
    #     # login for email
    #     sender = smtplib.SMTP("smtp.web.de", 587)
    #     sender.ehlo()
    #     sender.starttls()
    #     sender.ehlo()
    #     sender.login("user", "password")
    #
    #     mail = MIMEText(email)
    #     mail['Subject'] = game.config
    #     mail['From'] = "Me <emailaddress>"
    #     mail['To'] = "emailaddress"
    #
    #     sender.send_message(mail)
    #     sender.close()

    # delete current game history
    with open("bin/current_game_history.txt", "w") as file:
        file.write("")


if __name__ == '__main__':
    main()
