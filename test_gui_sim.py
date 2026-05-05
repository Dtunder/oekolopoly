import sys
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT_DIR = os.path.dirname(os.path.abspath("Final_Sovereign_Champion/oekolopoly_gui.py"))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import oekolopoly_gui

import pygame
pygame.init()
pygame.display.set_mode((800, 600))
class DummyArgs:
    def __init__(self):
        self.language = "en"

camera = oekolopoly_gui.Camera()
try:
    game = oekolopoly_gui.Game(camera, DummyArgs())
    print("POINTS INIT:", game.env.unwrapped.V[game.env.unwrapped.POINTS])
    guardian = oekolopoly_gui.SovereignGuardian(game.env)
    action, reason = guardian.get_final_action(int(game.env.unwrapped.V[game.env.unwrapped.POINTS]))
    print(action, reason)
    print("Type of returned action:", type(action))
    game.sovereign_move()
    print("Move executed!")
except Exception as e:
    print("ERROR:", e)
