# =====================================================================
# bubbles.py
#
# Bubbles
# 
#
# HISTORY:
#   Started: 2019-11-06 C Mason (CfA)
#
# =====================================================================
import matplotlib.pylab as plt
import numpy as np
import scipy
from scipy import interpolate, integrate

import os, glob, sys

from astropy.cosmology import Planck15
from astropy import units as u
from astropy import constants as const

import bubbles

# =====================================================================
# Constants

eV_tab    = np.logspace(np.log10(13.6), 4)*u.eV
nu_tab    = (eV_tab/const.h).to(u.Hz)
sigma_ion = 6.304e-18*u.cm**2. * (nu_tab / bubbles.nu_H)**-3.

# =====================================================================
# HII region size

def R_bubble_CenHaiman2000(z_s, Ndot_ion=2.e57/u.s, t_source=1e7*u.year):
    """
    Radius of HII region from Cen & Haiman 2000

    Neglects Hubble flow and recombinations, and assumes constant source output
    """
    R_bubble = (0.75*Ndot_ion*t_source/np.pi/bubbles.n_H(z_s))**(1./3.)
    return (R_bubble).to(u.Mpc)

def R_bubble_Stromgren(z_s, Ndot_ion=2.e57/u.s, C=3., alpha_B=2.59e-13*u.cm**3./u.s):
    """
    Stromgren sphere solution (ionization equilibrium)
    """
    return ((3.*Ndot_ion/4./np.pi/alpha_B/bubbles.n_H(z_s)**2./C)**(1./3.)).to(u.Mpc)

# =====================================================================
# Residual neutral fraction

def xHI_CenHaiman2000(z, z_s=7., C_HII=3., Ndot_ion=1.e57/u.s):
    """
    Neutral fraction in a HII region as a function of distance from it
    
    TODO: assumes quasar ionizing spectrum

    """
    r = bubbles.comoving_distance_from_source_Mpc(z, z_s)
    
    # Neutral fraction
    xHI = 1.e-6 * C_HII * (r/1/u.Mpc)**2. * (Ndot_ion/1.e57*u.s)**-1. * ((1.+z)/8.)**3.
    
    xHI[xHI > 1.] = 1.
    
    return xHI

def xHI_R(r, z_s, fesc=1., C=3., T=1e4, 
          J_bg=0., qso=True, alpha=-1.8):
    """
    Neutral fraction from source
    (Mesinger+04)
    
    for fesc = 0.5, Nion = 1e-57/s

    Args:
        J_bg (float): scaling of average UV background
    """    
    J_source_integrand = bubbles.L_nu(nu_tab, qso=qso, alpha=alpha) * sigma_ion/(bubbles.h_erg_s * nu_tab)
    Gamma12_source     = fesc/(4. * np.pi * r**2.) * np.trapz(J_source_integrand, nu_tab)
    Gamma12_background = J_bg * bubbles.Gamma12(z_s) / u.s

    xHI = C * bubbles.n_H(z_s) * bubbles.alpha_rec_B(T)/(Gamma12_background + Gamma12_source)
    
    return xHI.to(u.s/u.s)

def xHI_approx(xHI_01, Rtab, R_HII, r_slope=2.):
    """
    Approximate xHI as power law using a single value at 0.1 Mpc
    """
    xHI = xHI_01 * (10*Rtab.value)**r_slope
    xHI[Rtab > R_HII+0.001*u.Mpc] = 1.
    return xHI
    