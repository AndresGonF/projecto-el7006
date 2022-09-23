import torch.nn as nn
import pandas as pd
import numpy as np
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def random_dates(start='1970-01-01', end='2022-09-01', N=100):
    "Create N mjd within range."
    start_seconds = pd.to_datetime(start).value//10**9
    end_seconds = pd.to_datetime(end).value//10**9

    rand_dates = pd.to_datetime(np.random.randint(start_seconds, end_seconds, N), unit='s')
    rand_mjd = rand_dates.to_julian_date() - 2400000.5
    return rand_mjd.values.round(3)

class SaveBestModel:
    def __init__(self, best_val_loss=float('inf')):
        self.best_val_loss = best_val_loss
        self.best_model = None
        
    def __call__(self, current_val_loss, epoch, model):
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            # print(f"Best val loss: {self.best_val_loss}")
            # print(f"Saving best model for epoch: {epoch}\n")
            self.best_model = model