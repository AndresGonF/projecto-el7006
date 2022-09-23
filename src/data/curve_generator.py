from random import random
import numpy as np
from scipy import signal 
from src.utils import random_dates

def random_periodic_sin_mix(period, seq_len=100, s_noise=0.2, irregular=True):
    """Genera una curva aleatoria de suma de sinusoides.

    Parameters
    ----------
    seq_len : int
        Número int que define la cantidad de períodos a generar.
    s_noise : float
        float que fija la cantidad de ruido a agregar.

    Returns
    -------
    tuple
        tuple con los tiempos y magnitudes de la curva.            
    """
    mjd = random_dates(N=seq_len)
    if irregular:
        mjd += np.random.randn(seq_len)*0.1
    mjd = np.sort(mjd)
    mag = np.sin(2.0*np.pi*mjd/period) + 0.5*np.sin(2.0*np.pi*2*mjd/period)  + 0.25*np.sin(2.0*np.pi*3*mjd/period)
    mag += np.random.randn(seq_len)*s_noise
    return mjd, mag

def random_periodic_square_signal(period, seq_len=100, s_noise=0.2, irregular=True):
    """Genera una curva aleatoria de señales cuadradas.

    Parameters
    ----------
    seq_len : int
        Número int que define la cantidad de períodos a generar.
    s_noise : float
        float que fija la cantidad de ruido a agregar.

    Returns
    -------
    tuple
        tuple con los tiempos y magnitudes de la curva.            
    """    
    mjd = random_dates(N=seq_len)
    if irregular:
        mjd += np.random.randn(seq_len)*0.1
    mjd = np.sort(mjd)
    mag = signal.square(2 * np.pi * mjd/period) 
    mag += np.random.randn(seq_len)*s_noise
    return mjd, mag

def random_periodic_sawtooth_signal(period, seq_len=100, s_noise=0.2, irregular=True):
    """Genera una curva aleatoria de diente de sierra.

    Parameters
    ----------
    seq_len : int
        Número int que define la cantidad de períodos a generar.
    s_noise : float
        float que fija la cantidad de ruido a agregar.

    Returns
    -------
    tuple
        tuple con los tiempos y magnitudes de la curva.            
    """    
    mjd = random_dates(N=seq_len)
    if irregular:
        mjd += np.random.randn(seq_len)*0.1
    mjd = np.sort(mjd)
    mag = signal.sawtooth(2 * np.pi * mjd/period) 
    mag += np.random.randn(seq_len)*s_noise
    return mjd, mag


def random_gauss_signal(mu=0, sigma=1, seq_len=100, s_noise=0.2, irregular=True):
    """Genera una curva aleatoria de diente de sierra.

    Parameters
    ----------
    seq_len : int
        Número int que define la cantidad de períodos a generar.
    s_noise : float
        float que fija la cantidad de ruido a agregar.

    Returns
    -------
    tuple
        tuple con los tiempos y magnitudes de la curva.            
    """    
    mjd = random_dates(N=seq_len)
    if irregular:
        mjd += np.random.randn(seq_len)*0.1
    mjd = np.sort(mjd)
    mag = np.random.normal(loc=mu, scale=sigma, size=seq_len) 
    mag += np.random.randn(seq_len)*s_noise
    return mjd, mag    