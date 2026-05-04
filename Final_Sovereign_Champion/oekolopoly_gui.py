import os
import sys
import numpy as np
import gymnasium as gym
import pygame
import logging
from pygame.math import Vector2
from sb3_contrib import RecurrentPPO
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# WATERPROOF MONKEY PATCH for LSTM
import torch.nn as nn
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init

# Configure Professional Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SovereignChampion")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from oekolopoly.oekolopoly_gui.oeko_gui import OekoGui # We can reuse basic gui logic if we want, or do pure pygame
from translator import dict_translate, dict_help_screens
import oekolopoly.oekolopoly
from run_champion import SovereignGuardian

class GameStateManager:
    def __init__(self, language="en"):
        self.env = gym.make('Oekolopoly-v2', language=language)
        self.obs, _ = self.env.reset()
        
        # Load AI
        model_path = os.path.join(ROOT_DIR, "sota_recurrent_champion")
        if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
             logger.error(f"CRITICAL: Model file {model_path} not found!")
             sys.exit(1)
        self.agent = RecurrentPPO.load(model_path, device='cpu')
        self.guardian = SovereignGuardian(self.env)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        
        # State tracking
        self.done = False
        self.year = 0
        self.total_ap = 8
        self.ap_spent = [0]*6
        
    def step_ai(self):
        if self.done: return
        action, self.lstm_states = self.agent.predict(self.obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        final_action = self.guardian.get_final_action(action, int(self.env.unwrapped.V[9]))
        
        self.obs, reward, terminated, truncated, info = self.env.step(final_action)
        self.episode_starts = np.zeros((1,), dtype=bool)
        self.year = self.env.unwrapped.V[8]
        if terminated or truncated:
             self.done = True

class Button:
    def __init__(self, rect, color, text, font):
        self.rect = pygame.Rect(rect)
        self.color = color
        self.text = text
        self.font = font

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 2) # border
        img = self.font.render(self.text, True, (255, 255, 255))
        text_rect = img.get_rect(center=self.rect.center)
        surface.blit(img, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

class ModernGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1024, 768))
        pygame.display.set_caption("Sovereign Champion GUI (MVC Architecture)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.state = GameStateManager()
        self.running = True

        # Setup UI Elements
        self.btn_ai_step = Button((50, 50, 150, 50), (100, 100, 200), "AI Step", self.font)

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    if self.btn_ai_step.is_clicked(event.pos):
                         self.state.step_ai()

    def render_frame(self):
        self.screen.fill((30, 30, 30)) # Dark mode

        # Draw UI
        self.btn_ai_step.draw(self.screen)

        # Draw game state info
        V = self.state.env.unwrapped.V
        texts = [
            f"Year: {V[8]}",
            f"AP: {V[9]}",
            f"Sanitation: {V[0]}",
            f"Production: {V[1]}",
            f"Education: {V[2]}",
            f"Quality of Life: {V[3]}",
            f"Pop Growth: {V[4]}",
            f"Environment: {V[5]}",
            f"Population: {V[6]}"
        ]

        y_offset = 120
        for t in texts:
            img = self.font.render(t, True, (200, 200, 200))
            self.screen.blit(img, (50, y_offset))
            y_offset += 30

        if self.state.done:
             img = self.font.render("GAME OVER", True, (255, 100, 100))
             self.screen.blit(img, (300, 300))

        pygame.display.flip()

    def run(self):
        while self.running:
            self.process_events()
            self.render_frame()
            self.clock.tick(60)

if __name__ == "__main__":
    gui = ModernGUI()
    gui.run()
