import matplotlib.pyplot as plt
import numpy as np

def plot_periodic(mjd, mag, P, title, ax):
    ax[0].plot(mjd, mag, '.')
    ax[0].set_xlabel('Época')
    ax[0].set_ylabel('Magnitud')
    ax[0].set_title(f'{title} - Sin doblar')
     
    ax[1].plot(np.mod(mjd, P)/P, mag, '.')
    ax[1].set_xlabel('Fase')
    ax[1].set_ylabel('Magnitud')
    ax[0].set_title(f'{title} - Doblado')