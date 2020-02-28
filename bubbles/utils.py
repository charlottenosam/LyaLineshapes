# utils.py
"""Useful functions for blowing bubbles

History:
  Started: 2019-10-17 C Mason (CfA)


"""

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
import json
import itertools as it

from astropy.cosmology import Planck15
from astropy import units as u
from astropy import constants as const

import bubbles

# =====================================================================
# Constants
wave_Lya = 1216. * u.Angstrom
wave_em  = np.linspace(1210, 1220., 1000) * u.Angstrom

# =====================================================================

def n_H(z, X_p=0.75):
    """IGM hydrogen number density

    Args:
        z (ndarray): redshift

    Returns:
        number density in cm^-3

    Cen & Haiman 2000::
    >>> 8.5e-5 * ((1. + z)/8.)**3. / (u.cm**3.)


    """
    return (X_p * Planck15.Ob0 * Planck15.critical_density0 * (1+z)**3. / const.m_p).to(u.cm**(-3.))


def alpha_rec_B(T):
    """Recombination rate for `case B` recombination of hydrogen.
 
       Fitting formulae from Hui & Gnedin (1997) (in the Appendix)
       accurate to ~0.7% from 1 - 10^9K
     
        input T unitless (but in K)
       Returns recombination rate in cm**3/s
    """
    lHI = 2 * 157807.0 / T
    alpha_B = 2.753e-14 * lHI**1.5 / (1.0 + (lHI / 2.740)**0.407)**2.242

    return alpha_B * u.cm**3. / u.s

def alpha_rec_A(T):
    """Recombination rate for `case A` recombination of hydrogen.
 
       Fitting formulae from Hui & Gnedin (1997) (in the Appendix)
       accurate to ~2% from 3 - 10^9K
     
        input T unitless (but in K)

       Returns recombination rate in cm**3/s
    """
    lHI = 2 * 157807.0 / T
    alpha_A = 1.269e-13 * lHI**1.503 / (1.0 + (lHI / 0.522)**0.470)**1.923

    return alpha_A * u.cm**3. / u.s

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

def z_at_proper_distance(R_p, z_1=7.):
    R_H = (const.c / Planck15.H(z=z_1)).to(u.Mpc)
    R_com = R_p * (1+z_1)
    return z_1 - R_com/R_H

# def z_at_comoving_distance(R_com, z_1=7.):
#     R_H = (const.c / Planck15.H(z=z_1)).to(u.Mpc)
#     return z_1 - R_com/R_H

# ---------------------------------------------------------------------
# Emission lines
# ---------------------------------------------------------------------

# Wave to velocity offset
def wave_to_DV(wave):
    return ((wave - wave_Lya)*const.c/wave_Lya).to(u.km/u.s)

def DV_to_wave(DV):
    return wave_Lya + (DV/const.c * wave_Lya).to(u.Angstrom)

def make_DV_axes(ax_wave, x, y, DV_min=-1000., DV_max=1000.):
    # Make axis
    ax_DV = ax_wave.twiny()
    
    # Plot
    ax_DV.plot(x, y, lw=0)

    # Set xlims
    ax_DV.set_xlim(DV_min, DV_max)
    ax_wave.set_xlim(bubbles.DV_to_wave(np.array(ax_DV.get_xlim())*u.km/u.s).value)
    
    ax_DV.set_xlabel(r'Velocity offset, $\Delta v$ [km/s]')
    
    return ax_DV

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

# ---------------------------------------------------------------------

def scalar_mappable(array, cmap='plasma_r', rescale=0.):
    """
    Map from array to colors for plotting lines

    color = s_m.to_rgba(value)
    plt.colorbar(s_m)
    """
    # Normalise
    norm = mpl.colors.Normalize(vmin=(1-rescale)*np.min(array), vmax=(1+rescale)*np.max(array))

    # create ScalarMappable and initialize a data structure
    s_m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_m.set_array([])

    return s_m

def dict_to_image(dictionary, x, y):
    image = np.zeros((len(y), len(x)))
    for (i, yy), (j, xx) in it.product(enumerate(y), enumerate(x)):
        image[i, j] = dictionary[(xx,yy)]
        
    return image
# ---------------------------------------------------------------------