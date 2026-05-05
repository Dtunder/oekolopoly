import random
import uuid
import gym
import copy
from pygame.math import Vector2
import numpy as np
from stable_baselines3 import PPO
import smtplib
import datetime
import os
from email.mime.text import MIMEText

from oekolopoly import oekolopoly
import pygame

color_red = (255, 0, 0)
color_yellow = (255, 255, 0)
color_blue = (0, 0, 255)
color_white = (255, 255, 255)
color_black = (0, 0, 0)
color_turky = (83, 195, 189)
color_light_blue = (83, 255, 255)
color_green = (0, 255, 0)
color_brown = (185, 156, 107)
color_orange = (255, 185, 0)


# need to be outside main for real randomness
#random_number = random.randint(0, 2)
#user_id = uuid.uuid4()


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


def draw_text(pos, image, font_size, text, color=color_white):
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
    regions_act = distribute1(regions_act, env.V[env.POINTS])
    if reduce_production:
        regions_act[1] = -regions_act[1]

    for i in range(len(regions_act)):
        region_result = env.V[i] + regions_act[i]
        if region_result < env.Vmin[i]:
            regions_act[i] = env.Vmin[i] - env.V[i]
        elif region_result > env.Vmax[i]:
            regions_act[i] = env.Vmax[i] - env.V[i]

    act = np.append(regions_act, special_act)

    return act


class Camera(pygame.sprite.Group):
    def __init__(self):
        super().__init__()
        self.display_surface = pygame.display.get_surface()

        self.offset = Vector2()

        self.background = pygame.image.load('assets/Spielbrett_komplett.JPG').convert_alpha()
        self.background = pygame.transform.scale(self.background, self.display_surface.get_size())
        self.background_rect = self.background.get_rect(topleft=(0, 0))

        self.empty_background = pygame.image.load('assets/Spielbrett_komplett - Kopie.JPG').convert_alpha()
        self.empty_background = pygame.transform.scale(self.empty_background, self.display_surface.get_size())

        self.camera_movement_speed = 10
        self.zoom_factor = 1.0
        self.min_zoom = 1.0
        self.max_zoom = 3.0
        self.zoom_change = 0.1
        self.internal_surface = pygame.Surface(self.display_surface.get_size(), pygame.SRCALPHA)
        self.can_use_buttons = True
        self.show_background = True

    def custom_draw(self):
        self.internal_surface.fill(color_brown)

        offset = self.background_rect.topleft + self.offset
        if self.show_background:
            self.internal_surface.blit(self.background, offset)

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
    def __init__(self, camera):
        self.camera = camera
        self.env = gym.make('Oekolopoly-v1')
        self.agent_obs = self.env.reset()
        self.agent = PPO.load("trained_agents/obs_box_action_box_reward_perround_0.5_ppo_17.zip")
        self.current_action = [0, 0, 0, 0, 0, 0]
        self.all_actions = []
        self.action_font = pygame.font.SysFont('Times New Roman', 30)
        self.game_loop = True
        self.done = False
        self.available_actionpoints = self.env.V[self.env.POINTS]
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
                                          "Entferne verteilte Punkte", 20)
        self.step_button = Button(Vector2(600, 470), Vector2(175, 40), color_orange, camera, "Zug beenden")
        self.close_game = Button(Vector2(1670, 15), Vector2(245, 55), color_red, camera, "Spiel beenden", 40)
        self.reset_button = Button(Vector2(400, 350), Vector2(170, 40), color_red, camera, "Neues Spiel")
        self.predict_button = Button(Vector2(600, 410), Vector2(195, 40), color_turky, camera,
                                     f"Bester Zug ({self.predict_usages})")
        if self.max_predict_usages > 5:
            self.predict_button.text = "Bester Zug"
        self.preview_button = Button(Vector2(600, 350), Vector2(210, 40), color_green, camera, "Vorschaumodus")
        self.help_button = Button(Vector2(270, 455), Vector2(130, 70), color_yellow, camera, "Hilfe?", 40)
        self.game_instructions_button = Button(Vector2(20, 90), Vector2(190, 40), color_yellow, camera,
                                               "Spielanleitung")
        #self.feedback_button = Button(Vector2(20, 145), Vector2(145, 40), color_yellow, camera, "Feedback?")
        self.game_history_button = Button(Vector2(20, 145), Vector2(170, 40), color_yellow, camera, "Spielhistorie")

        # diagrams
        sanitation_d = Diagram(Vector2(940, 30), camera, self.env.Vmin[0], self.env.Vmax[0])
        production_d = Diagram(Vector2(1390, 30), camera, self.env.Vmin[1], self.env.Vmax[1])
        education_d = Diagram(Vector2(1745, 580), camera, self.env.Vmin[2], self.env.Vmax[2])
        quality_of_life_d = Diagram(Vector2(980, 730), camera, self.env.Vmin[3], self.env.Vmax[3])
        population_growth_d = Diagram(Vector2(570, 730), camera, self.env.Vmin[4], self.env.Vmax[4])
        environment_d = Diagram(Vector2(1680, 130), camera, self.env.Vmin[5], self.env.Vmax[5])
        population_d = Diagram(Vector2(170, 720), camera, self.env.Vmin[6], self.env.Vmax[6])
        politics_d = Diagram(Vector2(100, 320), camera, self.env.Vmin[7], self.env.Vmax[7])
        action_points_d = Diagram(Vector2(330, 70), camera, self.env.Vmin[9], self.env.Vmax[9])
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

        self.available_actionpoints_label = Label(Vector2(810, 410), Vector2(320, 40), camera, "Aktionspunkte übrig: ",
                                                  30, color_green, self.available_actionpoints)
        self.current_action_label = Label(Vector2(810, 470), Vector2(360, 40), camera, "Verteilte Punkte: ", 22,
                                          color_green)
        self.console_label = Label(Vector2(600, 530), Vector2(850, 30), camera, "", 20, color_yellow)
        self.preview_console_label = Label(Vector2(600, 580), Vector2(850, 30), camera, "", 20, color_yellow)
        self.round_label = Label(Vector2(420, 470), Vector2(140, 40), camera, "Runde: ", 30, color_green)
        self.game_over_label = Label(Vector2(860, 580), Vector2(250, 70), camera, "Game Over", 50, color_red)

        # region labels with more info labels #
        action_points_label = Label(Vector2(390, 48), Vector2(155, 40), camera, "Aktionspunkte", 23, color_white)
        more_info_action_points = MoreInfo(Vector2(390, 90), Vector2(170, 50),
                                           "Einfluss, Geld, Arbeit,\n"
                                           "Energie, Güter, Nahrung", camera)

        sanitation_label = Label(Vector2(815, 40), Vector2(115, 40), camera, "Sanierung", 23, color_white)
        more_info_sanitation = MoreInfo(Vector2(750, 82), Vector2(180, 105),
                                        "Umweltschutz, Recycling,\n"
                                        "Biotechnik, sanfte Energie,\n"
                                        "Landschaftserhaltung,\n"
                                        "Humanisierung der\n"
                                        "Arbeitswelt",
                                        camera)

        production_label = Label(Vector2(1250, 30), Vector2(122, 40), camera, "Produktion", 23, color_white)
        more_info_production = MoreInfo(Vector2(1227, 72), Vector2(145, 70),
                                        "Industrie, Handwerk,\n"
                                        "Landwirtschaft,\n"
                                        "Dienstleistungen", camera)

        environment_label = Label(Vector2(1725, 380), Vector2(180, 40), camera, "Umweltbelastung", 23, color_white)
        more_info_environment = MoreInfo(Vector2(1685, 422), Vector2(220, 105),
                                         "Abgase, Abwässer,\n"
                                         "Abwärme, Lärm, Raubbau,\n"
                                         "Landschaftszerstörung,\n"
                                         "Trennung natürlicher Kreisläufe,\n"
                                         "Verkehrschaos, Städtezerfall", camera)

        education_label = Label(Vector2(1790, 830), Vector2(125, 40), camera, "Aufklärung", 23, color_white)
        more_info_education = MoreInfo(Vector2(1675, 872), Vector2(240, 105),
                                       "Gesunde Lebensweise,\n"
                                       "Selbstverwirklichung, Schulen,\n"
                                       "Erwachsenenbildung, Umwelt-\n"
                                       "bewußtsein, sinnvolle Freizeit,\n"
                                       "Bürgerinitiativen, Geburtenkontrolle", camera)

        quality_of_life_label = Label(Vector2(1025, 980), Vector2(155, 40), camera, "Lebensqualität", 23, color_white)
        more_info_quality_of_life = MoreInfo(Vector2(1025, 908), Vector2(210, 70),
                                             "Gesundheit, Sicherheit,\n"
                                             "Sinnvolle Arbeit, Wohnqualität,\n"
                                             "Naherholung, Freizeitangebot", camera)

        population_growth_label = Label(Vector2(615, 980), Vector2(180, 40), camera, "Vermehrungsrate", 23, color_white)
        more_info_population_growth = MoreInfo(Vector2(615, 928), Vector2(210, 50),
                                               "Geburten, Sterbefälle,\n"
                                               "Unfälle, Zu- und Abwanderung", camera)

        population_label = Label(Vector2(215, 980), Vector2(140, 40), camera, "Bevölkerung", 23, color_white)
        more_info_population = MoreInfo(Vector2(215, 908), Vector2(240, 70),
                                        "Bevölkerungszahl, Menschendichte,\n"
                                        "Altersaufbau, Arbeitskräfte,\n"
                                        "Sozialstruktur", camera)

        politic_label = Label(Vector2(15, 435), Vector2(78, 40), camera, "Politik", 23, color_white)
        more_info_politic = MoreInfo(Vector2(15, 477), Vector2(90, 160),
                                     "Weitsicht,\n"
                                     "Autorität,\n"
                                     "Beliebtheit,\n"
                                     "einsichtige\n"
                                     "Programme,\n"
                                     "Ent-\n"
                                     "scheidungs-\n"
                                     "gewalt", camera)

        self.region_labels = (
            action_points_label, sanitation_label, production_label, environment_label, education_label,
            quality_of_life_label, population_growth_label, population_label, politic_label)
        self.more_info_labels = (
            more_info_action_points, more_info_sanitation, more_info_production, more_info_environment,
            more_info_education,
            more_info_quality_of_life, more_info_population_growth, more_info_population, more_info_politic)

        # special case labels and buttons
        self.special_population_growth_label = Label(Vector2(510, 640), Vector2(155, 35), camera, "Spezialfall", 30,
                                                     color_yellow)
        self.special_population_growth_decrease_button = Button(Vector2(510, 690), Vector2(35, 35), color_red, camera,
                                                                "-", 35)
        self.act_special_population_growth_label = Label(Vector2(570, 690), Vector2(35, 35), camera, "", 25,
                                                         color_yellow, "0")
        self.special_population_growth_increase_button = Button(Vector2(630, 690), Vector2(35, 35), color_green, camera,
                                                                "+", 35)
        # help screen
        self.help_screen = HelpScreen(camera)
        self.help_screen.active = get_number_value_by_name("bin/played_games.txt", "games_played") == 0
        self.help_screen.active = True

        self.toggle_game_features()

    def check_button_press(self):
        if not self.help_screen.active:
            if not self.done:
                for action_input_index in range(len(self.action_inputs)):
                    self.change_current_action(action_input_index,
                                               self.action_inputs[action_input_index].button_pressed())

                if self.step_button.button_pressed():
                    self.step_with_current_action()
                if self.clear_action_button.button_pressed():
                    self.reset_current_action()
                if self.preview_button.button_pressed():
                    self.preview_mode = not self.preview_mode
                if self.special_action:
                    self.use_special_action_button()

                if self.max_predict_usages > 5:
                    if self.predict_button.button_pressed():
                        self.predict_next_move()
                else:
                    if self.predict_button.button_pressed():
                        if not self.predict_used and self.predict_usages > 0:
                            self.predict_usages -= 1
                            self.predict_used = True
                            self.predict_button.text = f"Bester Zug ({self.predict_usages})"
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
        #if self.feedback_button.button_pressed():
        #    os.startfile("feedback.txt")
        if self.game_instructions_button.button_pressed():
            os.startfile("Spielanleitung.pdf")
        if self.game_history_button.button_pressed():
            os.startfile("current_game_history.txt")

    def change_current_action(self, action_index, change):
        if change > 0 and self.current_action[action_index] + 1 + self.env.V[action_index] <= self.env.Vmax[
            action_index]:
            if self.current_action[action_index] < 0:
                self.available_actionpoints += 1
                self.current_action[action_index] += 1
            elif self.available_actionpoints > 0:
                self.current_action[action_index] += 1
                self.available_actionpoints -= 1
        elif change < 0 and self.current_action[action_index] - 1 + self.env.V[action_index] >= self.env.Vmin[
            action_index]:
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
                self.current_action[5] -= 1
                self.special_action_points -= 1

    def update_diagrams(self):
        for diagram_index in range(len(self.diagrams)):
            env_position = diagram_index
            if env_position == 8:
                env_position += 1
            self.diagrams[diagram_index].current_value = self.env.V[env_position]

    def update_labels(self):
        self.round_label.variable_text = self.env.V[self.env.ROUND]
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

        if self.special_action:
            self.current_action_label.variable_text = self.current_action
            self.act_special_population_growth_label.variable_text = self.current_action[5]
        else:
            self.current_action_label.variable_text = self.current_action[0:5]

        for label_index in range(len(self.action_inputs)):
            number_change = self.current_action[label_index]
            if number_change > 0:
                text = f"+{number_change}"
            else:
                text = number_change
            self.action_inputs[label_index].current_label.variable_text = text

        self.update_preview_labels()

    def update(self):
        if self.camera.can_use_buttons:
            self.check_button_press()
            if not self.help_screen.active:
                self.checkHoverOnLabels()
        self.update_diagrams()
        self.update_labels()
        self.check_new_help()
        if self.preview_mode:
            self.preview_action()

    def predict_next_move(self):
        obs_for_agent = self.agent_obs + self.env.Vmin
        agent_action, _ = self.agent.predict(obs_for_agent, deterministic=True)
        a_for_env = transf_act_box(self.env, agent_action)
        self.current_action = a_for_env
        self.available_actionpoints = 0
        self.special_action_points = 5 - abs(self.current_action[5])

    def preview_action(self):
        temp_env = copy.deepcopy(self.env)
        action = list(self.current_action)

        action[temp_env.PRODUCTION] -= temp_env.Amin[temp_env.PRODUCTION]
        action[5] -= temp_env.Amin[5]
        _, _, done, info = temp_env.step_w_o_clip(action)
        for diagram_index in range(len(self.diagrams)):
            env_index = diagram_index
            if env_index == 8:
                env_index += 1
            self.diagrams[diagram_index].done = done
            if done:
                self.preview_console_label.variable_text = f"Vorsicht! Game Over mit aktuellem Zug: {info['done_reason']}"
            else:
                self.preview_console_label.variable_text = ""
            self.diagrams[diagram_index].next_value = temp_env.V[env_index]

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

    def step_with_current_action(self):
        if not self.done:
            self.console_label.text = ""
            action = list(self.current_action)

            # umwandlung zu zahlen > 0
            action[self.env.PRODUCTION] -= self.env.Amin[self.env.PRODUCTION]
            action[5] -= self.env.Amin[5]
            self.agent_obs, reward, done, info = self.env.step(action)

            if info['valid_move']:
                #if self.predict_used:
                #    self.all_actions.append(["ai"])
                #else:
                #    self.all_actions.append(["hu"])
                #if self.preview_mode:
                #    self.all_actions.append(["pm"])
                self.all_actions.append(self.current_action)
                self.reset_current_action()
                self.predict_used = False
                self.available_actionpoints = self.env.V[self.env.POINTS]
                self.special_action = self.env.V[self.env.EDUCATION] >= 20
                if done:
                    self.done = True
                    #games_played = get_number_value_by_name("bin/played_games.txt", "games_played")
                    #games_played += 1
                    #with open("bin/played_games.txt", "w") as file:
                    #   file.write(f"games_played: {games_played}")
                    self.console_label.text = f"{info['done_reason']}    Endergebnis: {round(reward)} Punkte"

                    #print(self.all_actions)
                    sani = ""
                    prod = ""
                    educ = ""
                    lifeq = ""
                    pgrow = ""
                    pgrowspec = ""
                    acts = [sani,prod,educ,lifeq,pgrow,pgrowspec]
                    roun = ""
                    round_counter = 0
                    for act in self.all_actions:
                        for i in range(len(acts)):
                            acts[i] += f"{act[i]}   "
                            if act[i] < 0 or act[i] > 9:
                                acts[i] = acts[i][:-1]
                            if act[i] < -9:
                                acts[i] = acts[i][:-1]
                        if round_counter < 9:
                            roun += f"{round_counter + 1}   "
                        else:
                            roun += f"{round_counter + 1}  "
                        round_counter += 1
                    current_game_text = (f"{self.env.V[8]} Runden ueberlebt, {round(reward)} Punkte erzielt\n"
                                         f"Ausgeführte Züge in Runde:    {roun}\n"
                                         f"Sanierung:                    {acts[0]} Sanierung\n"
                                         f"Produktion:                   {acts[1]} Produktion\n"
                                         f"Aufklärung:                   {acts[2]} Aufklärung\n"
                                         f"Lebensqualität:               {acts[3]} Lebensqualität\n"
                                         f"Vermehrungsrate:              {acts[4]} Vermehrungsrate\n"
                                         f"Vermehrungsratespezialfall:   {acts[5]} Vermehrungsratespezialfall\n\n\n")
                    #print(current_game_text)
                    with open("current_game_history.txt", "a") as history:
                        history.write(current_game_text)

                    #date_now = datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
                    #text = f"Runde: {self.env.V[8]}, Punkte: {round(reward)}, Spiele beendet: {games_played}, Zustaende: {self.env.V}, Abbruchgrund: {info['done_reason']}, Datum: {date_now}, Zuege: {self.all_actions}\n"
                    #with open("bin/game_history.txt", "a") as history:
                    #    history.write(text)
            else:
                self.console_label.text = info['invalid_move_info']

    def reset_current_action(self):
        self.current_action = [0, 0, 0, 0, 0, 0]
        self.available_actionpoints = self.env.V[self.env.POINTS]
        self.special_action_points = 5

    def reset_game(self):
        #if not self.done and self.all_actions != []:
        #date_now = datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        #text = f"Vorzeitiger Abbruch des Spiels, Runde: {self.env.V[8]}, Zustaende {self.env.V}, Datum: {date_now}, Zuege: {self.all_actions}\n"
        #with open("bin/game_history.txt", "a") as history:
        #    history.write(text)

        self.env = gym.make('Oekolopoly-v1')
        self.agent_obs = self.env.reset()
        self.all_actions = []
        self.current_action = [0, 0, 0, 0, 0, 0]
        self.console_label.text = ""
        self.available_actionpoints = self.env.V[self.env.POINTS]
        self.done = False
        self.special_action = False
        self.special_action_points = 5
        #self.preview_mode = False
        self.predict_usages = self.max_predict_usages
        self.predict_used = False
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
        self.image.blit(self.button_text_font.render(f"{self.text}", True, color_black),
                        (10 / 1920 * pygame.display.get_window_size()[0], text_position))
        pygame.draw.rect(self.image, color_black, pygame.Rect((0, 0), self.size), 2)

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
        self.image.blit(self.text_font.render(f"{self.text}{self.variable_text}", True, color_black),
                        (10 / 1920 * pygame.display.get_window_size()[0], text_position))
        pygame.draw.rect(self.image, color_black, pygame.Rect((0, 0), self.size), 2)


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
            self.preview_label.rect.topleft = self.preview_label_scaled_pos + Vector2(0, 135 / 1080 *
                                                                                      pygame.display.get_window_size()[
                                                                                          1])
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

            # diagram boarder
            pygame.draw.rect(self.image, color_blue, pygame.Rect(0, 0, diagram_width, diagram_height), boarder_size)

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
    def __init__(self, camera):
        super().__init__(camera)
        self.image = pygame.Surface(pygame.display.get_window_size()).convert_alpha()
        self.image.fill((0, 0, 0, 0))
        self.rect = self.image.get_rect()
        self.rect.topleft = Vector2()
        self.active = False
        self.camera = camera
        self.scale_x = pygame.display.get_window_size()[0] / 1920
        self.scale_y = pygame.display.get_window_size()[1] / 1080
        self.exit_help_button = Button(Vector2(910, 850), Vector2(135, 40), color_red, camera, "Verlassen")
        self.help_mode = Label(Vector2(860, 900), Vector2(260, 80), camera, "Hilfemodus", 50, color_yellow)
        self.step_back_button = Button(Vector2(760, 1000), Vector2(140, 40), color_green, camera, "<- Zurück")
        self.page_label = Label(Vector2(910, 1000), Vector2(160, 40), camera, "Seite: ", 30, color_turky)
        self.step_forward_button = Button(Vector2(1080, 1000), Vector2(140, 40), color_green, camera, "Weiter ->")

        # special case labels and buttons as preview if not unlocked yet
        self.special_label = Label(Vector2(510, 640), Vector2(155, 35), camera, "Spezialfall", 30, color_yellow)
        self.special_negative_label = Label(Vector2(510, 690), Vector2(35, 35), camera, "-", 35, color_red)
        self.special_number_label = Label(Vector2(570, 690), Vector2(35, 35), camera, "", 25, color_yellow, "0")
        self.special_positive_label = Label(Vector2(630, 690), Vector2(35, 35), camera, "+", 35, color_green)
        self.special_case_picture = pygame.image.load('assets/education_population_growth.jpg').convert_alpha()
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
            self.step_forward_button.text = "Verlassen"
            self.step_forward_button.color = color_red
        else:
            self.step_forward_button.text = "Weiter ->"
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
        #extra_large_font_size = int(50 * x)
        if self.active:
            # grey background
            pygame.draw.rect(self.image, (0, 0, 0, 200), pygame.Rect(0, 0, 1920 * x, 1080 * y))
            pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(1670 * x, 15 * y, 245 * x, 55 * y))
            #played_games = get_number_value_by_name("bin/played_games.txt", "games_played")

            if self.help_step == 1:
                draw_text(Vector2(400 * x, 200 * y), self.image, big_font_size,
                          "Willkommen bei Ökolopoly!\n"
                          "Deine Aufgabe ist es ein Land zu führen!\n\n"
                          "Du befindest dich gerade im Hilfemodus.\n"
                          "Hier wird das Wichtigste zum Spielen von\n"
                          "Ökolopoly und den Features kurz erklärt.\n\n"
                          "Ziel des Spiels ist es erstmal so viele Runden\n"
                          "wie möglich zu überleben, da man erst nach 10 Runden\n"
                          "Punkte verdienen kann. Dann gilt die Punktzahl!\n\n"
                          #"Zuerst werden 3 Spiele ohne neue Features gespielt.\n"
                          #"Verdeckte Features werden nach 3 zu Ende gespielten\n"
                          #"('Game over') Spielen je nach Gruppe freigeschaltet.\n\n"
                          #"Es wäre schön wenn mindestens 5 Spiele beendet werden.\n"
                          #f"Bereits beendete Spiele: {played_games}\n\n"
                          #"Bitte zum Beenden des Spiels den 'Spiel beenden'-Knopf\n"
                          #"verwenden, damit die Daten anonymisiert übermittelt\n"
                          #"werden können!\n\n"
                          "Viel Erfolg!")
            elif self.help_step == 2:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(805 * x, 15 * y, 280 * x, 330 * y))
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(805 * x, 405 * y, 610 * x, 110 * y))
                draw_text(Vector2(470 * x, 15 * y), self.image, small_font_size,
                          "Es gibt 9 Bereiche auf die man\n"
                          "achten sollte, dass diese nicht zu\n"
                          "niedrig oder zu hoch werden")
                draw_text(Vector2(470 * x, 110 * y), self.image, small_font_size,
                          "Davon lassen sich 5 Bereiche\n"
                          "steuern. Die Zahl zwischen dem\n"
                          "+ und - kann man mit den\n"
                          "Knöpfen anpassen. Dieser Wert\n"
                          "bestimmt am Ende einer Runde\n"
                          "wie der Bereich verändert\n"
                          "werden soll\n\n"
                          "Hinweis:\n"
                          "Produktion akzeptiert als\n"
                          "einziger Bereich auch negative\n"
                          "Werte als Veränderung!")
                draw_text(Vector2(1100 * x, 20 * y), self.image, small_font_size,
                          "Der Bereich gibt mehr Infos,\n"
                          "wenn man mit der Maus darüber fährt.")
                draw_text(Vector2(1100 * x, 140 * y), self.image, small_font_size,
                          "Immer rechts neben dem Balken steht\n"
                          "der aktuelle Wert des Bereiches.")
                draw_text(Vector2(805 * x, 520 * y), self.image, small_font_size,
                          "Zu Beginn jeder Runde wird eine bestimme Anzahl an\n"
                          "Aktionspunkten bereitgestellt. Diese kann man beliebig auf\n"
                          "die 5 Bereiche verteilen.\n"
                          "Mit dem Knopf entferne verteilte Punkte werden alle in\n"
                          "dieser Runde verteilten Aktionspunkte zurückgesetzt.")
            elif self.help_step == 3:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(415 * x, 465 * y, 150 * x, 50 * y))
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(395 * x, 345 * y, 180 * x, 50 * y))
                draw_text(Vector2(585 * x, 455 * y), self.image, big_font_size,
                          "Die Runde zeigt an in welchem Jahr\n"
                          "man sich befindet. Es beginnt bei 0")
                draw_text(Vector2(585 * x, 300 * y), self.image, small_font_size,
                          "Mit dem Knopf Neues Spiel kann\n"
                          "ein neues Spiel gestartet werden.\n"
                          "Der bisherige Fortschritt geht dabei\n"
                          "aber verloren!")
            elif self.help_step == 4:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(595 * x, 465 * y, 580 * x, 50 * y))
                draw_text(Vector2(595 * x, 520 * y), self.image, small_font_size,
                          "Mit dem Knopf Zug beenden schließt man das aktuelle\n"
                          "Jahr mit den verteilten Aktionspunkten ab.\n"
                          "Einige der Bereiche haben Auswirkungen auf andere und diese\n"
                          "Auswirkungen werden mit Abschluss des Jahres eintreten!")
            elif self.help_step == 5:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(505 * x, 635 * y, 165 * x, 95 * y))
                self.image.blit(self.special_case_picture, (790 * x, 200 * y))
                draw_text(Vector2(680 * x, 635 * y), self.image, small_font_size,
                          "Der Spezialfall tritt ein wenn die Aufklärung einen Wert von 20 oder höher erreicht hat.\n"
                          "Nun kann man die Vermehrungsrate an einem weiteren Ort anpassen.\n"
                          "Der Vorteil hierbei:\n"
                          "Es werden keine Aktionspunkte benötigt! Dafür lässt es sich nur von -5 bis +5 verändern.\n"
                          "Achtung! Bei einer Aufklärung von 20 haben nur Werte von -3 bis 3 eine Auswirkung!\n"
                          "Für niedrigere/höhere Werte muss die Aufklärung höher sein! Siehe Grafik.")
            elif self.help_step == 6:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(595 * x, 525 * y, 860 * x, 40 * y))
                draw_text(Vector2(595 * x, 570 * y), self.image, small_font_size,
                          "Das ist die Konsole. Hier wird nach dem Ende eines Spiels Feedback gegeben:\n"
                          "Warum das Spiel zu Ende ist und wie viele Punkte man erzielt hat.")
            elif self.help_step == 7:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(15 * x, 85 * y, 200 * x, 105 * y))
                draw_text(Vector2(225 * x, 60 * y), self.image, small_font_size,
                          "Mit dem Knopf Spielanleitung öffnet sich die originale\n"
                          "Spielanleitung des Spiels Ökolopoly.\n\n"
                          "Mit dem Knopf Spielhistorie kann man sich die bereits gespielten Spiele anschauen.\n\n\n\n"
                          #"Mit dem Knopf Feedback öffnet sich ein Textdokument in\n"
                          #"das man eine kurze Rückmeldung zum Spiel schreiben kann.\n"
                          #"Bitte die Textdatei speichern!\n\n\n\n"
                          "Mit den Tasten WASD oder den Pfeiltasten auf der Tastatur\n"
                          "kann man sich frei auf dem Spielbrett bewegen und mit dem\n"
                          "Mausrad hinein- und herauszoomen.\n"
                          "Beim maximalen Herauszoomen zentriert sich das Spielbrett automatisch.\n"
                          "Mit der Tabulatortaste lassen sich die Spielelemente ein- und ausblenden.\n\n"
                          "Achtung:\n"
                          "Spielelemente lassen sich nur beim maximalen herausgezoomten Zustand bedienen.")
            elif self.help_step == 8:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(805 * x, 15 * y, 280 * x, 330 * y))
                draw_text(Vector2(1100 * x, 70 * y), self.image, small_font_size,
                          "Das Balken Feature gibt Auskunft über die\n"
                          "minimalen und maximalen Werte die jeder Bereich\n"
                          "annehmen darf.\n\n"
                          "Die Farbe vom momentanen Wert wird\n"
                          "zur Mitte Grün und zu den beiden\n"
                          "Grenzwerten Rot.")
                #s = ""
                #s2 = "s"
                #if self.max_help_steps == 10:
                #    features = "Balken Feature, Vorschaumodus und Bester Zug."
                #elif self.max_help_steps == 9:
                #    features = "Balken Feature und Vorschaumodus."
                #else:
                #    features = "Balken Feature."
                #    s = "s"
                #    s2 = ""
                #if played_games == 3:
                #    draw_text(Vector2(500 * x, 500 * y), self.image, extra_large_font_size,
                #              f"Hurra du hast folgende{s} neue{s} Feature{s2} freigeschaltet!\n"
                #              f"{features}\n\n"
                #              f"Dazu gibt es neue Hilfeseiten!", color_green)
            elif self.help_step == 9:
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(595 * x, 345 * y, 220 * x, 50 * y))
                draw_text(Vector2(370 * x, 280 * y), self.image, small_font_size,
                          "Das ist der Vorschaumodusknopf\n"
                          "Damit schaltet man den Vorschaumodus\n"
                          "an und aus.")
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(805 * x, 15 * y, 280 * x, 330 * y))
                pygame.draw.rect(self.image, color_black, pygame.Rect(941 * x, 280 * y, 34 * x, 3 * y))
                draw_text(Vector2(1095 * x, 50 * y), self.image, small_font_size,
                          "Der Vorschaumodus zeigt an wie sich die Zustände mit den\n"
                          "aktuell verteilten Aktionspunkten im nächsten Jahr verändern\n"
                          "würden.\n"
                          "Eine schwarze Markierung zeigt die Veränderung des Balken an.\n"
                          "Außerdem wird mithilfe einer Zahl über beziehungsweise\n"
                          "bei negativen Zahlen unter dem aktuellen Wert die Abnahme/Zunahme\n"
                          "des Bereiches angezeigt!\n\n"
                          "Hinweis:"
                          "Sind diese Grün überlebt man das Jahr. Sind diese jedoch\n"
                          "Orange verliert man wenn man den Zug beendet.\n"
                          "Die Konsole gibt Auskunft dazu!")
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(595 * x, 575 * y, 860 * x, 40 * y))
                draw_text(Vector2(595 * x, 620 * y), self.image, small_font_size,
                          "Das ist die Konsole vom Vorschaumodus. Diese gibt Auskunft über den Grund\n"
                          "des Spielendes, sollte man den aktuellen Zug durchführen wollen.")
            elif self.help_step == 10:
                max_predict_uses = get_number_value_by_name("bin/config.txt", "predict")
                text = ("Der Knopf Bester Zug verteilt die verfügbaren Aktionspunkte so gut\n"
                        "es geht sinnvoll auf die Bereiche.\n"
                        "Dahinter steckt ein Agent(KI), der darauf trainiert wurde.\n\n")
                if max_predict_uses <= 5:
                    text += f"Achtung!\nDie Nutzung ist auf {max_predict_uses} mal pro Spiel begrenzt!"
                pygame.draw.rect(self.image, (0, 0, 0, 0), pygame.Rect(595 * x, 405 * y, 205 * x, 50 * y))
                draw_text(Vector2(595 * x, 460 * y), self.image, small_font_size, text)


def main():
    #setup_game(random_number, user_id)

    fps = 60
    pygame.init()
    pygame.display.set_mode()
    #pygame.display.set_mode((1920, 1080))
    pygame.display.set_caption("Ökolopoly")
    clock = pygame.time.Clock()

    camera = Camera()
    game = Game(camera)

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

        # update game
        game.update()
        if not game.game_loop:
            game_loop = False

        #print(pygame.mouse.get_pos())

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
    feedback_text = get_text_from_file("feedback.txt")
    if len(feedback_text) > 0:
        email += f"\nFeedback\n {feedback_text}"
        with open("feedback.txt", "w") as file:
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
    with open("current_game_history.txt", "w") as file:
        file.write("")


if __name__ == '__main__':
    main()
