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
# Ionizing background model Khaire & Srianand 2019
# [Khaire & Srianand 2019](https://arxiv.org/abs/1801.09693)

KS2019  = np.genfromtxt(bubbles.base_path+'bubbles/parameters_Q18.txt', skip_header=4)
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


def Muv_to_Nion(Muv, z, alpha_s=-2., beta=-2):
    """
    Convert Muv to Nion [s^-1]
    """

    Lnu_912 = Muv_to_Lnu(Muv, z, beta=beta)
    Nion    = Lnu_912/const.h/-alpha_s
    return Nion.to(1/u.s)


def Muv_to_Lnu(Muv, z, beta=-2., Kcorr=False):
    """
    Convert UV magnitude to L_912 (nu)
    """

    lum_dist = Planck15.luminosity_distance(z)

    if Kcorr:
        # k-correction http://adsabs.harvard.edu/full/2000A%26A...353..861W
        K_corr = (beta + 1) * 2.5 * np.log10(1.0 + z)
    else:
        K_corr = 0.

    # Apparent mag
    mab = Muv + 5.0 * (np.log10(lum_dist.to(u.pc).value) - 1.0) + K_corr

    f0      = 3.631E-20*u.erg/u.s/u.Hz/u.cm**2.
    c       = 3.E5*u.km/u.s
    wave912 = 912.*u.Angstrom

    fnu_1500 = f0 * 10**(-0.4*mab)
    fnu_912  = fnu_1500 * (wave912/1500/u.Angstrom)**(beta+2.)
    Lnu_912  = fnu_912 * 4*np.pi * lum_dist**2.

    return Lnu_912.to(u.erg/u.s/u.Hz)
