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
        self.btn_ai_step = Button((50, 50, 150, 50), (60, 60, 120), "AI Step", self.font)

        # Soft animation tracking
        self.warning_alpha = 0
        self.warning_fade_in = True

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    if self.btn_ai_step.is_clicked(event.pos):
                         self.state.step_ai()

    def render_frame(self):
        # High-tech background
        self.screen.fill((20, 22, 25))

        # Draw UI
        self.btn_ai_step.draw(self.screen)

        # Draw game state info with smooth design
        V = self.state.env.unwrapped.V
        texts = [
            f"Year: {V[8]:02d}",
            f"Action Points: {V[9]:02d}",
            f"Sanitation: {V[0]:02d}",
            f"Production: {V[1]:02d}",
            f"Education: {V[2]:02d}",
            f"Quality of Life: {V[3]:02d}",
            f"Pop Growth: {V[4]:02d}",
            f"Environment: {V[5]:02d}",
            f"Population: {V[6]:02d}",
            f"Politics: {V[7]:02d}"
        ]

        y_offset = 120
        # Draw High-Tech Sliders instead of just text
        for i, t in enumerate(texts):
            # Text label
            img = self.font.render(t, True, (180, 180, 200))
            self.screen.blit(img, (50, y_offset))

            # Draw progress bar/slider for the specific V values
            if i >= 2 and i <= 9: # Skip Year and AP
                 val = V[i-2]
                 max_val = 29 if i-2 < 6 else 48 if i-2 == 6 else 37 # rough maxes
                 bar_width = 150
                 # Smooth interpolation could go here, but discrete bar is fine
                 fill_width = max(0, min(bar_width, int((val / max_val) * bar_width)))
                 pygame.draw.rect(self.screen, (50, 50, 60), (250, y_offset + 8, bar_width, 10), border_radius=5)
                 # Color logic based on health
                 bar_color = (100, 255, 100)
                 if i-2 == 5 and val <= 12: bar_color = (255, 100, 100) # Env bad
                 if i-2 == 7 and val < 0: bar_color = (255, 100, 100) # Pol bad

                 pygame.draw.rect(self.screen, bar_color, (250, y_offset + 8, fill_width, 10), border_radius=5)

            y_offset += 35

        # Draw AI Reasoning Logs
        y_offset = 120
        log_header = self.font.render("AI Reasoning Logs:", True, (100, 200, 255))
        self.screen.blit(log_header, (450, 80))

        reasoning_texts = []
        if V[5] <= 9 or V[7] < -5:
            reasoning_texts.append("[Black Sky] Emergency Production Override")
        if V[3] >= 28:
            reasoning_texts.append("[Critical] Emergency QoL Cut")

        if len(reasoning_texts) == 0:
            reasoning_texts.append("> Standard Homeostasis Maintained")

        for t in reasoning_texts:
            img = self.font.render(t, True, (150, 255, 150) if "Standard" in t else (255, 150, 150))
            self.screen.blit(img, (450, y_offset))
            y_offset += 30

        # Blinking Warning Lights if Politics or Env are critical
        if V[7] < 0 or V[5] < 12:
             if self.warning_fade_in:
                  self.warning_alpha += 5
                  if self.warning_alpha >= 200: self.warning_fade_in = False
             else:
                  self.warning_alpha -= 5
                  if self.warning_alpha <= 50: self.warning_fade_in = True

             warning_surface = pygame.Surface((1024, 768), pygame.SRCALPHA)
             pygame.draw.rect(warning_surface, (255, 0, 0, self.warning_alpha), (0, 0, 1024, 768), 5)
             self.screen.blit(warning_surface, (0,0))

             warn_text = self.font.render("SYSTEM WARNING: SUB-OPTIMAL EQUILIBRIUM", True, (255, 50, 50))
             self.screen.blit(warn_text, (450, 30))

        if self.state.done:
             end_text = "SIMULATION TERMINATED" if V[8] < 30 else "UTOPIA SECURED (30 Years Reached)"
             color = (255, 100, 100) if V[8] < 30 else (100, 255, 100)

             # Draw a nice overlay box
             overlay = pygame.Surface((400, 100), pygame.SRCALPHA)
             overlay.fill((20, 20, 20, 200))
             self.screen.blit(overlay, (300, 420))
             pygame.draw.rect(self.screen, color, (300, 420, 400, 100), 2)

             img = self.font.render(end_text, True, color)
             self.screen.blit(img, (320, 450))

        pygame.display.flip()

    def run(self):
        while self.running:
            self.process_events()
            self.render_frame()
            self.clock.tick(60)

if __name__ == "__main__":
    gui = ModernGUI()
    gui.run()
