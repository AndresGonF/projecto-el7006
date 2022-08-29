import matplotlib.pyplot as plt
import numpy as np

def plot_periodic(mjd, mag, P, ax):
    ax[0].plot(mjd, mag, '.')
    ax[0].set_xlabel('Tiempo')
    ax[1].plot(np.mod(mjd, P)/P, mag, '.')
    ax[1].set_xlabel('Fase');