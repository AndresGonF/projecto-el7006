import numpy as np
from scipy import signal 

def random_periodic_light_curve(period, N=100, s_noise=0.2):

    mjd = np.linspace(0, 4, num=N, dtype=np.float32)
    mjd += np.random.randn(N)*0.1
    mjd = np.sort(mjd)
    mag = np.sin(2.0*np.pi*mjd/period) + 0.5*np.sin(2.0*np.pi*2*mjd/period)  + 0.25*np.sin(2.0*np.pi*3*mjd/period)
    mag += np.random.randn(N)*s_noise
    return mjd, mag

def random_periodic_square_signal(period, N=100, s_noise=0.2):
    mjd = np.linspace(0, 4, num=N, dtype=np.float32)
    mjd += np.random.randn(N)*0.1
    mjd = np.sort(mjd)
    mag = signal.square(2 * np.pi * mjd/period) 
    mag += np.random.randn(N)*s_noise
    return mjd, mag

def random_periodic_triang_signal(period, N=100, s_noise=0.2):
    mjd = np.linspace(0, 4, num=N, dtype=np.float32)
    mjd += np.random.randn(N)*0.1
    mjd = np.sort(mjd)
    mag = signal.sawtooth(2 * np.pi * mjd/period) 
    mag += np.random.randn(N)*s_noise
    return mjd, mag