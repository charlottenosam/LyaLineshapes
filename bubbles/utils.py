# =====================================================================
# utils.py
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

# =====================================================================

def n_H(z):
    """
    IGM hydrogen density
    """
    return 8.5e-5 * ((1. + z)/8.)**3. / (u.cm**3.)

# ---------------------------------------------------------------------
# Cosmology, distances
# ---------------------------------------------------------------------
def dt_dz(z):
    return 1. / ((1. + z) * Planck15.H(z))

def comoving_distance_from_source_Mpc(z_2, z_1):
    """
    COMOVING distance between z_1 and z_2
    """
    R_com = (z_1 - z_2)*(const.c / Planck15.H(z=z_1)).to(u.Mpc)
    return R_com

def z_at_comoving_distance(R_com, z_1=7.):
    R_H = (const.c / Planck15.H(z=z_1)).to(u.Mpc)
    return z_1 - R_com/R_H

# Wave to velocity offset
def wave_to_DV(wave):
    return ((wave - wave_Lya)*const.c/wave_Lya).to(u.km/u.s)

def DV_to_wave(DV):
    return wave_Lya + (DV/const.c * wave_Lya).to(u.Angstrom)



