# =====================================================================
# lya_cross_section.py
#
# Useful functions for blowing bubbles
#
# HISTORY:
#   Started: 2019-10-17 C Mason (CfA)
#
# =====================================================================
import matplotlib
from functools import reduce
import matplotlib.pylab as plt
import numpy as np
import math
import scipy
import scipy.interpolate
from sklearn.neighbors import KernelDensity
import os, glob

from astropy.cosmology import Planck15
from astropy import units as u
from astropy import constants as const

import bubbles

# =====================================================================
# Constants
wave_Lya = 1216. * u.Angstrom
freq_Lya = (const.c / wave_Lya).to(u.Hz)

# =====================================================================

def av(T):
    try:
        T = T.value
    except:
        pass
    return 4.7e-4 * (T/1.e4)**-0.5

def Voigt(x, T=1.e4):    
    phix = np.exp(-x**2.) + av(T)/np.sqrt(np.pi)/x**2.
    phix[phix > 1.] = 1.
    return phix

def v_thermal(T=1.e4*u.K):
    return (1./np.sqrt(const.m_p/2./const.k_B/T)).to(u.km/u.s)

def Lya_wave_to_x(wave, T=1.e4*u.K):
    freq = (const.c / wave).to(u.Hz)
    
    v_therm   = v_thermal(T)
    dfreq_Lya = (freq_Lya * v_therm / const.c).to(u.Hz)
    
    # Dimensionless frequency
    x = (freq - freq_Lya)/dfreq_Lya
    
    return x

def Lya_x_to_wave(x, T=1.e4*u.K):
    
    v_therm   = v_thermal(T)
    dfreq_Lya = (freq_Lya * v_therm / const.c).to(u.Hz)
    
    # Dimensionless frequency
    wave = const.c / (x*dfreq_Lya + freq_Lya)
    
    return wave.to(u.Angstrom)

def Lya_crosssec(wave, T=1.e4*u.K, returnx=False):
    
    x = Lya_wave_to_x(wave, T)
    
    sig_Lya0 = 5.9e-14 * (T/1.e4/u.K)**-0.5
    sig_Lya  = sig_Lya0 * Voigt(x, T) * u.cm**2.
    
    if returnx:
        return x, sig_Lya
    else:
        return sig_Lya

# =====================================================================

def optical_depth_Weinberg97(z, n_HI):
    """
    Lya optical depth from https://ui.adsabs.harvard.edu/abs/1997ApJ...490..564W/abstract Eqn 3

    for uniform IGM

    tau = πe^2/(m_e c) * f_alpha lambda_alpha * n_HI/H(z)
    """
    tau = 1.34e-17*u.cm**3/u.s * Planck15.H(z).to(1./u.s) * n_HI

    return tau.value