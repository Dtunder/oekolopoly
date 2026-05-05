import sys
import os
sys.path.append(os.path.abspath('Final_Sovereign_Champion'))

# LSTM monkey patch
import torch.nn as nn
original_lstm_init = nn.LSTM.__init__
def patched_lstm_init(self, input_size, hidden_size, *args, **kwargs):
    return original_lstm_init(self, int(input_size), int(hidden_size), *args, **kwargs)
nn.LSTM.__init__ = patched_lstm_init

from sb3_contrib import RecurrentPPO

model = RecurrentPPO.load("Final_Sovereign_Champion/sota_recurrent_champion.zip", device="cpu")
print("Obs:", model.observation_space)
print("Act:", model.action_space)
