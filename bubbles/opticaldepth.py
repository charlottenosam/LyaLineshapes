# =====================================================================
# opticaldepth.py
#
# Optical depths
# 
#
# HISTORY:
#   Started: 2019-11-14 C Mason (CfA)
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

def optical_depth_grid(wave_em, T, z_min, z_max, z_s=7.,
                  inside_HII=True, C_HII=3., 
                  xHI_01=1e-8, R_ion=1.*u.Mpc,
                  r_slope=2.):
    """
    Lya optical depth as a function of wavelength 
    using definition of optical depth and Lya cross-section
    """
    crosssec = bubbles.lya_cross_section(T)   
    
    # Redshift array
    ztab_ends = np.array([z_min, z_max])
    
    # Observed wavelength
    wave_obs = wave_em * (1. + z_s)
        
    # Range of redshifted wavelength and x
    wave_z_ends = wave_obs[:,None]/(1+ztab_ends)
    x_z_ends    = crosssec.Lya_wave_to_x(wave_z_ends)
    
    tau = np.zeros(len(wave_obs))
    for ww, w_obs in enumerate(wave_obs):
                
        # Make xtab 
        if (x_z_ends[ww] < 0).all():
            xtab = -np.logspace(np.log10(-x_z_ends[ww].min()),np.log10(-x_z_ends[ww].max()),100)
            xtab = np.sort(xtab)
        elif (x_z_ends[ww] > 0).all():
            xtab = np.logspace(np.log10(x_z_ends[ww].min()),np.log10(x_z_ends[ww].max()),100)
            xtab = np.sort(xtab)
        else:       
            xtab_neg = -np.logspace(-1,np.log10(-x_z_ends[ww].min()),50)
            xtab_pos = np.logspace(-1,np.log10(x_z_ends[ww].max()),50)
            xtab     = np.sort(np.concatenate((xtab_neg, xtab_pos)))
        
        # Get wave_redshift
        wave_redshift = crosssec.Lya_x_to_wave(xtab)

        # Get z tab
        ztab = w_obs/wave_redshift - 1.

        # Residual neutral fraction
        if inside_HII:
            r   = bubbles.comoving_distance_from_source_Mpc(ztab, z_s)
            xHI = bubbles.xHI_approx(xHI_01, r, R_ion, r_slope=r_slope)        
        else:
            xHI = 1.
            
        # Cross-section
        lya_cross = crosssec.Lya_crosssec_x(xtab)
                           
        # Calculate optical depth
        prefac = (const.c * bubbles.dt_dz(ztab) * xHI * bubbles.n_H(ztab)).to(1./u.cm**2.)
        dtau   = prefac * lya_cross
            
        tau[ww] = np.trapz(dtau, ztab)
        
    return tau

def make_tau_grid(R_ion, xHI_01, r_slope=2., z_s=7., z_min=6.):
    """
    Make tau_HII, tau_IGM, tau_total for grid of R_ion, xHI(r=0.1 Mpc)
    """
    z_ion = bubbles.z_at_comoving_distance(R_ion, z_1=z_s)

    # inside bubble
    tau_HII = optical_depth_grid(bubbles.wave_em, z_min=z_ion, z_max=z_s, z_s=z_s,
                            inside_HII=True, T=1.e4*u.K,
                            xHI_01=xHI_01, R_ion=R_ion, r_slope=r_slope)

    # in IGM
    tau_IGM = optical_depth_grid(bubbles.wave_em, z_min=z_min, z_max=z_ion, z_s=z_s,
                            inside_HII=False, T=1.*u.K,
                            xHI_01=xHI_01, R_ion=R_ion)

    tau_total = tau_IGM + tau_HII
    
    tau_tab = tau_HII, tau_IGM, tau_total
    return tau_tab

