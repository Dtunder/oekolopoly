import sys
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import oekolopoly_gui

import pygame

def test_gui_sim():
    pygame.init()
    pygame.display.set_mode((800, 600))
    class DummyArgs:
        def __init__(self):
            self.language = "en"

    camera = oekolopoly_gui.Camera()
    try:
        game = oekolopoly_gui.Game(camera, DummyArgs())
        game.sovereign_move()
        assert True
    except Exception as e:
        assert False, f"GUI sim failed: {e}"

if __name__ == '__main__':
    test_gui_sim()
