import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from astropy.time import Time
import astropy.units as u

def calc_phase_ecc(times, Tp, Torb, e):
        """Returns the phase and works for an eccentric orbit."""
        phases = np.array( ( ( (times - Tp) ) / Torb)%1 ).flatten()
        phases = convert_ma_to_ta(phases, e)

        return phases
    
def convert_ma_to_ta(ma, e):
        """Return conversion of Mean Anomaly to True Anomaly."""
        phases = np.arange(0,1.00001,0.00001)
        f = interp1d(mean_anomaly(phases, e), phases, bounds_error=False, fill_value='extrapolate')

        return f(ma)
    
def mean_anomaly(true_anomaly, e):
        """Return the Mean Anomaly."""
        ta = true_anomaly*np.pi*2
        ma = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(ta/2)) + [2*np.pi if p>np.pi else 0 for p in ta] - e*(1-e**2)*np.sin(ta)/(1+e*np.cos(ta))

        return ma / (np.pi*2)