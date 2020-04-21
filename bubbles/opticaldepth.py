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
    x_z_ends    = crosssec.Lya_wave_to_x(wave_z_ends).value
    
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
            r_com = bubbles.comoving_distance_from_source_Mpc(ztab, z_s)
            r_p   = r_com / (1+z_s)
            xHI   = C_HII * bubbles.xHI_approx(xHI_01, r_p, R_ion, r_slope=r_slope)        
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
    z_ion = bubbles.z_at_proper_distance(R_ion, z_1=z_s)

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


# ------------------------------------------------------------

def optical_depth(wave_em, T, z_min, z_max, z_s=7.,
                  inside_HII=True, C_HII=3., xtab_len=100,
                  Ndot_ion=1.e57/u.s):
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
    x_z_ends    = crosssec.Lya_wave_to_x(wave_z_ends).value
    
    tau = np.zeros(len(wave_obs))
    for ww, w_obs in enumerate(wave_obs):
                
        # Make xtab 
        if (x_z_ends[ww] < 0).all():
            xtab = -np.logspace(np.log10(-x_z_ends[ww].min()),np.log10(-x_z_ends[ww].max()),xtab_len)
            xtab = np.sort(xtab)
        elif (x_z_ends[ww] > 0).all():
            xtab = np.logspace(np.log10(x_z_ends[ww].min()),np.log10(x_z_ends[ww].max()),xtab_len)
            xtab = np.sort(xtab)
        else:       
            xtab_neg = -np.logspace(-1,np.log10(-x_z_ends[ww].min()),int(xtab_len/2))
            xtab_pos = np.logspace(-1,np.log10(x_z_ends[ww].max()),int(xtab_len/2))
            xtab     = np.sort(np.concatenate((xtab_neg, xtab_pos)))
        
        # Get wave_redshift
        wave_redshift = crosssec.Lya_x_to_wave(xtab)

        # Get z tab
        ztab = w_obs/wave_redshift - 1.

        # Residual neutral fraction
        if inside_HII:
            r_com = bubbles.comoving_distance_from_source_Mpc(ztab, z_s)
            r_p   = r_com / (1+z_s)
            xHI   = bubbles.xHI_R(r=r_p, z_s=z_s, Ndot_ion=Ndot_ion, fesc=1., J_bg=1., C=C_HII, T=T.value)    
        else:
            xHI = 1.
            
        # Cross-section
        lya_cross = crosssec.Lya_crosssec_x(xtab)
                           
        # Calculate optical depth
        prefac = (const.c * bubbles.dt_dz(ztab) * xHI * bubbles.n_H(ztab)).to(1./u.cm**2.)
        dtau   = prefac * lya_cross
    
        tau[ww] = np.trapz(dtau, ztab)
        
    return tau


def make_tau(Ndot_ion_total, source_age, wave_em, z_s=7., z_min=6., C=3, R_type='CH00'):
    """Make optical depth given Nion and source age
    """

    if R_type == 'CH00':
        R_ion = bubbles.R_bubble_CenHaiman2000(z_s=z_s, Ndot_ion=Ndot_ion_total, t_source=source_age)
    else:
        # Find full ionized bubble
        R3 = scipy.integrate.odeint(bubbles.dRion3_dt, y0=0., t=[0, source_age.value], args=(z_s, Ndot_ion_total, C), tfirst=True)
        R_ion = R3.T[0][1]**(1/3.) * u.Mpc

    z_ion = bubbles.z_at_proper_distance(R_ion, z_1=z_s)

    # inside bubble
    tau_HII = optical_depth(wave_em, z_min=z_ion, z_max=z_s, z_s=z_s, 
                            inside_HII=True, T=1.e4*u.K, C_HII=C,Ndot_ion=Ndot_ion_total)

    # in IGM
    tau_IGM = optical_depth(wave_em, z_min=z_min, z_max=z_ion, z_s=z_s,
                            inside_HII=False, T=1.*u.K, C_HII=C, Ndot_ion=Ndot_ion_total)

    tau_total = tau_IGM + tau_HII
    
    tau_tab = tau_HII, tau_IGM, tau_total

    return tau_tab, R_ion


def plot_tau(tau_tab, wave_em, R_ion, transmission=False, vlim=1000,
             ax=None, annotate=True, annotation=None, 
             lw=2, ls='solid', label=None):
    """Plot optical depths
    """

    if len(tau_tab) == 3:
        tau_HII, tau_IGM, tau_total = tau_tab
    
    if transmission:
        if len(tau_tab) == 3:
            tau_HII, tau_IGM, tau_total = np.exp(-tau_HII), np.exp(-tau_IGM), np.exp(-tau_total)
        else:
            tau_tab = np.exp(-tau_tab)
            
        ylabel = r'Ly$\alpha$ transmission, $e^{-\tau_{\mathrm{Ly}\alpha}}$'
    else:
        ylabel = r'Optical depth $\tau_{\mathrm{Ly}\alpha}$'
    
    # Make DV table
    DV_tab = bubbles.wave_to_DV(wave_em)

    if ax is None:
        fig, ax_wave = plt.subplots(1,1)
    else:
        ax_wave=ax

    # DV plot
    ax_DV = ax_wave.twiny()
    
    if len(tau_tab) == 3:
        ax_wave.plot(wave_em, tau_HII, lw=1, ls='dashed', label='inside HII region')
        ax_wave.plot(wave_em, tau_IGM, lw=1, ls='dotted', label='IGM')
        ax_wave.plot(wave_em, tau_total, label='total')

        ax_DV.semilogy(DV_tab, tau_total, label='total', lw=0)
    
    else:
        ax_wave.plot(wave_em, tau_tab, lw=lw, ls=ls, label=label)

        ax_DV.semilogy(DV_tab, tau_tab, lw=0)        

    ax_wave.legend(loc='upper right')#, frameon=True)
    
    if transmission:
        plt.yscale('linear')
        ax_wave.set_ylim(-0.1, 1.1)
    else:
        plt.yscale('log')
        ax_wave.set_ylim(1e-6, 1.2e6)
        
    ax_DV.set_xlim(-vlim, vlim)
    ax_wave.set_xlim(bubbles.DV_to_wave(np.array(ax_DV.get_xlim())*u.km/u.s).value)

    if annotate:
        if annotation is None:
            annotation = '$R_{HII}=%.1f$ pMpc' % R_ion.value
        else:
            annotation = annotation
        ax_wave.annotate(annotation, xy=(0.98, 0.05), xycoords='axes fraction', ha='right', backgroundcolor='w')
    
    ax_wave.set_ylabel(ylabel)
    ax_wave.set_xlabel('Wavelength [A]')
    ax_DV.set_xlabel('Velocity offset [km/s]')
    
    plt.tight_layout()
    
    return

def Ix_ME1998(x):
	I = x**4.5/(1-x) + 9./7*x**3.5 + 9./5*x**2.5 + 3*x**1.5 + 9*x**0.5 - 4.5*np.log((1+x**0.5)/(1-x**0.5))
	return I