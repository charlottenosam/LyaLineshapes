# =====================================================================
# ionizing_background.py
#
# Ionizing sources model
# 
# --------------------
# Ionizing background model Khaire & Srianand 2019
#
# [Khaire & Srianand 2019](https://arxiv.org/abs/1801.09693) 
# fiducial quasar SED $\alpha=-1.8$. 
# 
# Which was found to reproduce the measured He II Lyman-α effective optical 
# depths as a function of z and the epoch of He II reionization
# --------------------
#
# HISTORY:
#   Started: 2019-10-17 C Mason (CfA)
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
nu_H    = (13.6*u.eV/const.h).to(u.Hz)
h_erg_s = (const.h).to(u.erg * u.s)

# =====================================================================

# ---------------------------------------
# Ionizing background model

KS2019  = np.genfromtxt('../data/KS_2018_EBL/parameters_Q18.txt', skip_header=4)
Gamma12 = interpolate.interp1d(KS2019[:,0], KS2019[:,1])

def plot_Gamma12_bg():

	plt.figure()

	plt.plot(KS2019[:,0], KS2019[:,1], label='Khaire & Srianand 2019')

	plt.legend()
	plt.xlabel('Redshift, z')
	plt.ylabel('Ionizing background, $\Gamma_{HI}$ [s$^{-1}$]')

# ---------------------------------------
# Ionizing spectrum

def L_nu(nu, qso=True, alpha=-1.8):
    """
    in erg/s/Hz
    """
    if qso:
        L_nu = 2.34e31 * (nu/nu_H)**alpha
    return L_nu * u.erg/u.s/u.Hz

def Ndot_ion_from_Lnu(nu, fesc=1., qso=True, alpha=-1.8):
    integrand = L_nu(nu, qso=qso, alpha=alpha) /(h_erg_s * nu)
    return fesc * np.trapz(integrand, nu).to(1/u.s)


