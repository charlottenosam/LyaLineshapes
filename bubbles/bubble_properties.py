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

from astropy.cosmology import Planck15, z_at_value
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
    r_com = bubbles.comoving_distance_from_source_Mpc(z, z_s)
    r_p   = r_com / (1+z_s)

    # Neutral fraction
    xHI = 1.e-6 * C_HII * (r_p/1/u.Mpc)**2. * (Ndot_ion/1.e57*u.s)**-1. * ((1.+z)/8.)**3.
    
    xHI[xHI > 1.] = 1.
    
    return xHI

def xHI_R(r, z_s, fesc=1., C=3., T=1e4, 
          J_bg=0., qso=True, alpha=-1.8):
    """
    R is proper distance

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
    
#===============================
# Solving ODE

def ionization_front_ionizations(z, Ndot_ion=1.e57/u.s):
    return (0.75 * Ndot_ion / (np.pi * bubbles.n_H(z))).to(u.Mpc**3./u.yr)

def ionization_front_recombinations(z, R=1.*u.Mpc, C=3, T=1e4):
    a_B = bubbles.alpha_rec_B(T)
    return (-a_B * C * bubbles.n_H(z) * R**3.).to(u.Mpc**3./u.yr)

def ionization_front_Hubble(z, R=1.*u.Mpc):
    return (3* Planck15.H(z) * R**3.).to(u.Mpc**3./u.yr)

def dRion3_dt(t, R3, z_s0=10, Ndot_ion=1.e55/u.s,
            C=3, T=1e4):
  #         Ndot_ion_type='constant', Ndot_ion_alpha=-2.):
    """
    Rate of growth of R^3 ionization front
    in Mpc/yr
    
    NB: for solve_ivp must be fun(t, y)
    """
    # Age of universe when source turned on
    t_source_on = Planck15.age(z=z_s0)
    
    # At age of source, t, what is redshift?
    t_now = t*u.yr + t_source_on
    if type(t_now.value) is float:
        z_tab_sourceage = z_at_value(Planck15.age, t_now)
    else:   
        z_tab_sourceage = np.array([z_at_value(Planck15.age, time) for time in t_now])
     
    # try:
    #     z_tab_sourceage = np.array([z_at_value(Planck15.age, time) for time in t_now])
    # except:
    #     z_tab_sourceage = z_at_value(Planck15.age, t_now)

    R = R3**(1./3.) * u.Mpc
    
#     # Ndot_ion
#     if Ndot_ion
#     Ndot_ion_burst(t, burst_freq=5e7*u.yr, burst_height=1.e57/u.s,
                   
#     print(t, Ndot_ion)           
    # Components of ionization front growth
    # All in Mpc**3/yr
    dR3dt_ion = ionization_front_ionizations(z_s0, Ndot_ion=Ndot_ion).value
    dR3dt_rec = ionization_front_recombinations(z_s0, R=R, C=C, T=T).value
    dR3dt_Hub = ionization_front_Hubble(z_tab_sourceage, R=R).value
    
    return dR3dt_ion - dR3dt_rec + dR3dt_Hub

# =============================
# Proximity zone

def R_optically_thin(z, Ndot_ion, alpha_s, reccase='B', 
                     T=1e4*u.K, fesc=1, C=3, tau_lim=2.3, J_bg=0):
    """
    Proper radius of Lya optically thin region
    
    aka proximity zone
    """
    
    A = 1.34e-7 * u.cm**3. / u.s
    sigma_ion0 = 6.3e-18 * u.cm**2
    
    if reccase == 'B':
        alpha_rec = bubbles.alpha_rec_B(T.value)
    else:
        alpha_rec = bubbles.alpha_rec_A(T.value)

    gamma_lim = A * C**2. * bubbles.n_H(z)**2. * alpha_rec / Planck15.H(z) / tau_lim
    
    R_alpha2 = fesc/4./np.pi * sigma_ion0 * Ndot_ion * (alpha_s/(alpha_s - 3)) / (gamma_lim - J_bg*bubbles.Gamma12(z)/u.s)
    
    return np.sqrt(R_alpha2).to(u.Mpc)


def Muv_to_Lnu(Muv, z, beta=-2.):
    """
    Convert UV magnitude to L_912
    """
    
    lum_dist = Planck15.luminosity_distance(z)
    mab = Muv + 5.0 * (np.log10(lum_dist.to(u.pc).value) - 1.0) - (2.5 * np.log10(1.0 + z))
    
    f0      = 3.631E-20*u.erg/u.s/u.Hz/u.cm**2.
    c       = 3.E5*u.km/u.s
    wave912 = 912.*u.Angstrom
    
    fnu_912 = f0 * 10**(-0.4*mab) * (1500*u.Angstrom/wave912)**(beta-2.)
    Lnu_912 = fnu_912 * 4*np.pi * lum_dist**2.
    
    return Lnu_912.to(u.erg/u.s/u.Hz)


def Muv_to_Nion(Muv, z, alpha_s=-1.8, beta=-2):
    """
    Convert Muv to Nion [s^-1]
    """
    
    Lnu_912 = Muv_to_Lnu(Muv, z, beta=beta)
    Nion = Lnu_912/const.h/-alpha_s
    return Nion.to(1/u.s)


def blue_velocity_lim(R_alpha, z):
    """
    Calculate velocity offset at a given proper radius
    """
    vlim = R_alpha * Planck15.H(z) #/(1+z)
    return vlim.to(u.km/u.s)