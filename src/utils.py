import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import copy
from pathlib import Path


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def random_dates(start='1970-01-01', end='2022-09-01', N=100, irregular=True):
    "Create N mjd within range."
    start_seconds = pd.to_datetime(start).value//10**9
    end_seconds = pd.to_datetime(end).value//10**9

    if irregular:
        rand_dates = pd.to_datetime(np.random.randint(start_seconds, end_seconds, N), unit='s')
    else:
        rand_dates =  pd.to_datetime(np.linspace(start_seconds, end_seconds, N), unit='s')
    rand_mjd = rand_dates.to_julian_date() - 2400000.5
    return rand_mjd.values.round(3) 

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def fix_seq_length(mag, mjd, seq_len):
    mag, mjd = lc_downsampling(mag, mjd, seq_len)
    mag, mjd = lc_padding(mag, mjd, seq_len)
    return mag, mjd

def lc_downsampling(mag, mjd, seq_len):
    data_lengths = np.array([seq.shape[0] for seq in mag])
    idx = np.arange(mag.shape[0])
    idx_geq = idx[data_lengths > seq_len]
    idx_geq_seq = [np.random.choice(lenghts, seq_len) for lenghts in data_lengths[idx_geq]]
    mag[idx_geq] = [mag_seq[idx_geq_seq[idx]] for idx, mag_seq in enumerate(mag[idx_geq])]
    mjd[idx_geq] = [mjd_seq[idx_geq_seq[idx]] for idx, mjd_seq in enumerate(mjd[idx_geq])]
    return mag, mjd

def symmetry_padding(sequence, wrap_length, phase=False):
    n = int(wrap_length / sequence.shape[0])
    mod = wrap_length % sequence.shape[0]
    if mod == 0:
        x0 = np.concatenate([sequence] * n)
    else:
        x0 = np.concatenate([sequence] * n + [sequence[:mod]])
    return x0

def lc_padding(mag, mjd, seq_len):
    data_lengths = np.array([seq.shape[0] for seq in mag])
    idx = np.arange(mag.shape[0])
    idx_leq = idx[data_lengths < seq_len]
    mag[idx_leq] = [symmetry_padding(mag_seq, seq_len) for mag_seq in mag[idx_leq]]
    mjd[idx_leq] = [symmetry_padding(mjd_seq, seq_len) for mjd_seq in mjd[idx_leq]]
    return mag, mjd

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