import os
import sys
import numpy as np
import gymnasium as gym
import pygame
import logging
from pygame.math import Vector2

# Performance & Stability Environment Setup
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Configure Professional Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SovereignChampion")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from translator import dict_translate, dict_help_screens
from summary_generator import SummaryGenerator
import oekolopoly.oekolopoly
from run_champion import SovereignGuardian

class GameStateManager:
    def __init__(self, language="en", use_ai=False):
        self.env = gym.make('Oekolopoly-v2', language=language)
        self.obs, _ = self.env.reset()
        self.use_ai = use_ai
        
        # Load AI only if requested (bypasses hangs on Py3.14)
        self.agent = None
        self.lstm_states = None
        if self.use_ai:
            try:
                import torch
                import torch.nn as nn
                from sb3_contrib import RecurrentPPO

                # WATERPROOF MONKEY PATCH for LSTM
                original_lstm_init = nn.LSTM.__init__
                def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
                    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
                nn.LSTM.__init__ = patched_lstm_init

                torch.set_num_threads(1)
                torch.set_grad_enabled(False)

                model_path = os.path.join(ROOT_DIR, "sota_recurrent_champion")
                self.agent = RecurrentPPO.load(model_path, device='cpu')
                self.episode_starts = np.ones((1,), dtype=bool)
            except Exception as e:
                logger.warning(f"AI Model could not be loaded: {e}. Falling back to Heuristic Zen Logic.")
                self.use_ai = False

        self.guardian = SovereignGuardian(self.env)

        # State tracking
        self.done = False
        self.year = 0
        self.reasoning = "Initializing Sovereign Champion..."
        self.state_history = [self.env.unwrapped.V.copy()]
        
    def step_ai(self):
        if self.done: return
        
        V = self.env.unwrapped.V
        avail = int(V[9])

        if self.use_ai and self.agent:
            # Full Hybrid mode
            action, self.lstm_states = self.agent.predict(self.obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
            self.episode_starts = np.zeros((1,), dtype=bool)
            final_action, reasons = self.guardian.get_final_action(action, avail)
        else:
            # Pure Heuristic Zen Logic (V290)
            final_action, reasons = self.guardian.get_final_action(None, avail)

        self.reasoning = ", ".join(reasons) if reasons else "Stability maintained."

        self.obs, reward, terminated, truncated, info = self.env.step(final_action)
        self.year = self.env.unwrapped.V[8]
        self.state_history.append(self.env.unwrapped.V.copy())
        
        if terminated or truncated:
             self.done = True
             self.reasoning = f"Simulation ended: {info.get('done_reason')}"

             # Generate Summary
             try:
                 generator = SummaryGenerator(self.state_history, dict_translate.get("en", {}))
                 generator.generate_svg(os.path.join(ROOT_DIR, "reports", "simulation_summary.svg"))
                 logger.info("Simulation summary SVG generated in /reports folder.")
             except Exception as e:
                 logger.error(f"Failed to generate summary: {e}")

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
    def __init__(self, use_ai=False):
        pygame.init()
        self.screen = pygame.display.set_mode((1024, 768))
        pygame.display.set_caption("Sovereign Champion GUI (V2.1 - Analytics Integrated)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.state = GameStateManager(use_ai=use_ai)
        self.running = True

        # Setup UI Elements
        self.btn_ai_step = Button((50, 50, 150, 50), (60, 60, 120), "AI Step", self.font)

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    if self.btn_ai_step.is_clicked(event.pos):
                          self.state.step_ai()

    def render_frame(self):
        self.screen.fill((15, 20, 28)) # Premium Deep Dark Mode

        # Draw UI
        self.btn_ai_step.draw(self.screen)

        # Draw reasoning (V290 Brain)
        reason_img = self.font.render(f"Log: {self.state.reasoning}", True, (100, 200, 255))
        self.screen.blit(reason_img, (50, 680))

        # Draw game state info with icons-like colors
        V = self.state.env.unwrapped.V
        texts = [
            (f"Year: {int(V[8])}", (255, 215, 0)),
            (f"AP: {int(V[9])}", (50, 255, 50)),
            ("-" * 20, (100, 100, 100)),
            (f"Sanitation: {int(V[0])}", (200, 200, 200)),
            (f"Production: {int(V[1])}", (255, 100, 100)),
            (f"Education: {int(V[2])}", (100, 100, 255)),
            (f"Quality of Life: {int(V[3])}", (255, 255, 100)),
            (f"Pop Growth: {int(V[4])}", (255, 150, 50)),
            (f"Environment: {int(V[5])}", (100, 255, 100)),
            (f"Population: {int(V[6])}", (200, 200, 200)),
            (f"Politics: {int(V[7])}", (255, 50, 50))
        ]

        y_offset = 120
        for t, color in texts:
            img = self.font.render(t, True, color)
            self.screen.blit(img, (50, y_offset))
            y_offset += 35

        if self.state.done:
             overlay = pygame.Surface((1024, 768), pygame.SRCALPHA)
             tint = (255, 0, 0, 50) if V[8] < 30 else (0, 255, 0, 50)
             overlay.fill(tint)
             self.screen.blit(overlay, (0,0))

             status_text = "SUCCESS (30 Years Reached)" if V[8] >= 30 else f"TERMINATED: {self.state.reasoning}"
             img = self.font.render(status_text, True, (255, 255, 255))
             rect = img.get_rect(center=(512, 384))
             self.screen.blit(img, rect)

             if V[8] >= 30:
                 sub_text = "Report generated in /reports folder."
                 img2 = self.font.render(sub_text, True, (200, 200, 200))
                 rect2 = img2.get_rect(center=(512, 430))
                 self.screen.blit(img2, rect2)

        pygame.display.flip()

    def run(self):
        while self.running:
            self.process_events()
            self.render_frame()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    gui = ModernGUI(use_ai=False)
    gui.run()
