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

class lya_cross_section(object):
    """
    Make Lya cross-sectin given temperature
    """

    def __init__(self, T=1.e4*u.K):
        """
        
        """
        self.T = T

        # Calculate thermal velocity
        self.v_therm   = self.v_thermal()
        self.dfreq_Lya = (freq_Lya * self.v_therm / const.c).to(u.Hz)

        # Voigt parameter
        self.av_T = self.av()

        # Cross-section peak
        self.sig_Lya0 = 5.9e-14 * (self.T/1.e4/u.K)**-0.5

        return

    def av(self):
        return 4.7e-4 * (self.T.value/1.e4)**(-0.5)

    def Voigt_badapprox(self, x): 
        """Clumsy approximation for Voigt
        """   
        phix = np.exp(-x**2.) + self.av_T/np.sqrt(np.pi)/x**2.
        phix[phix > 1.] = 1.
        return phix


    def Voigt(self, x):
        """Voigt function approximation from Tasitsiomi 2006
        
        https://ui.adsabs.harvard.edu/abs/2006ApJ...645..792T/abstract
        
        Good to >1% for T>2K. Correctly normalized to return H(av, x).

        int(phix)  = 1
        int(Voigt) = sqrt(pi)
        
        Args:
            x (ndarray): dimensionless frequency
        
        Returns:
            Voigt function
        """
        
        z = (x**2. - 0.855)/(x**2. + 3.42)
        
        q = z * (1 + 21./x**2.) * self.av_T/np.pi/(x**2. + 1) * \
             (0.1117 + z*(4.421 + z*(-9.207 + 5.674*z)))
        
        phix = np.exp(-x**2.)/1.77245385 
        
        phix[z > 0] += q[z > 0]

        return phix * np.sqrt(np.pi)


    def v_thermal(self):
        """
        Thermal velocity (Maxwell-Boltzmann)
        """
        return (1./np.sqrt(const.m_p/2./const.k_B/self.T)).to(u.km/u.s)

    def Lya_wave_to_x(self, wave):
        """
        Convert wavelength to x
        """
        freq = (const.c / wave).to(u.Hz)
                
        # Dimensionless frequency
        x = (freq - freq_Lya)/self.dfreq_Lya
        
        return x

    def Lya_x_to_wave(self, x):
        """
        Convert x to wavelength
        """                
        # Wavelength
        wave = const.c / (x*self.dfreq_Lya + freq_Lya)
        
        return wave.to(u.Angstrom)

    def Lya_crosssec(self, wave, returnx=False):
        
        x = self.Lya_wave_to_x(wave)
        
        sig_Lya  = self.sig_Lya0 * self.Voigt(x) * u.cm**2.
        
        if returnx:
            return x, sig_Lya
        else:
            return sig_Lya
    
    def Lya_crosssec_x(self, x):
                
        sig_Lya  = self.sig_Lya0 * self.Voigt(x) * u.cm**2.
        
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