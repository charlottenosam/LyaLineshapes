# =====================================================================
# utils.py
#
# Useful functions for blowing bubbles
#
# HISTORY:
#   Started: 2019-10-17 C Mason (CfA)
#
# =====================================================================
import matplotlib as mpl
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

def alpha_rec_B(T):
    """Recombination rate for `case B` recombination of hydrogen.
 
       Fitting formulae from Hui & Gnedin (1997) (in the Appendix)
       accurate to ~2% from 3 - 10^9K
     
       Returns recombination rate in cm**3/s
    """
    lHI = 2 * 157807.0 / T
    alpha_B = 2.753e-14 * lHI**1.5 / (1.0 + (lHI / 2.740)**0.407)**2.242

    return alpha_B * u.cm**3. / u.s

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

# ---------------------------------------------------------------------
# Emission lines
# ---------------------------------------------------------------------

# Wave to velocity offset
def wave_to_DV(wave):
    return ((wave - wave_Lya)*const.c/wave_Lya).to(u.km/u.s)

def DV_to_wave(DV):
    return wave_Lya + (DV/const.c * wave_Lya).to(u.Angstrom)


def lineshape_doublepeak(v, vcenter):
    """
    Gaussian line shape
    
    :param v:        velocity offset from systemic
    :param vcenter:  center of gaussian in v

    :return: Gaussian lineshape as a function of v
    """
    fwhm  = vcenter
    sigma = fwhm/2.355
    
    if vcenter == 0:
        sigma = 10/2.355
        line = 0.3989422804014327 * np.exp(-0.5*((v + vcenter)/sigma)**2.) / sigma
    else:
        line_blue = 0.3989422804014327 * np.exp(-0.5*((v + vcenter)/sigma)**2.) / sigma
        line_red  = 0.3989422804014327 * np.exp(-0.5*((v - vcenter)/sigma)**2.) / sigma
        line  = line_blue + line_red

    # Truncate and normalise
    line /= np.trapz(line, v)
    return line


def scalar_mappable(array, cmap='plasma_r'):
    """
    Map from array to colors for plotting lines

    color = s_m.to_rgba(value)
    plt.colorbar(s_m)
    """
    # Normalise
    norm = mpl.colors.Normalize(vmin=np.min(array), vmax=np.max(array))

    # create ScalarMappable and initialize a data structure
    s_m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_m.set_array([])

    return s_m