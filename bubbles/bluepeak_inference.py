# bluepeak_inference.py
"""Inference functions for blue peaks

History:
  Started: 2020-04-14 C Mason (CfA)


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

# Dynesty imports
import pickle
import dynesty
from dynesty import utils as dyfunc

import bubbles

# =====================================================================
# Constants

# =====================================================================


def lnlike(theta, vlim, sigma_v, Muv, Muv_err, z, fix_bg):
    """
    Likelihood for observing vlim+sigma_v given theta
    
    Likelihood is that R_alpha > vlim/H(z) --> erfc
    
    """
    
    if fix_bg is True:
        fesc, C, alpha_s, beta = theta
        J_bg = 0.
    
    else:
        fesc, C, alpha_s, beta, gamma_bg_12 = theta    
        J_bg = gamma_bg_12 * 1e-12 / bubbles.Gamma12(z)
    
    
    R_obs     = np.abs(vlim / Planck15.H(z)).to(u.Mpc)
    R_obs_err = np.abs(sigma_v / Planck15.H(z)).to(u.Mpc)

    # Ionizing photon flux
    Muv_draw = np.random.normal(loc=Muv, scale=Muv_err)
    Ndot_ion = bubbles.Muv_to_Nion(Muv_draw, z, alpha_s=alpha_s, beta=beta)
    
    # Model proximity zone
    R_mod = bubbles.R_optically_thin(z, Ndot_ion=Ndot_ion, alpha_s=alpha_s, 
                                     fesc=fesc, C=C,
                                     reccase='B', T=1e4*u.K, tau_lim=2.3, 
                                     J_bg=J_bg)
    
    if np.isnan(R_mod.value):
        R_mod = np.inf * u.Mpc
    
    likelihood = 0.5 * scipy.special.erfc((R_obs - R_mod) / np.sqrt(2.) / R_obs_err)

    return np.log(likelihood)


def lnprior(theta, fix_bg):
    """For emcee"""
    if fix_bg:
        fesc, C, alpha_s, beta = theta
        
        if 0. <= fesc <= 1. and 0.2 < C < 10. \
        and -2.5 <= alpha_s <= -1. \
        and -3 < beta < -1:
            return 0.
        else:
            return -np.inf
        
    else:
        fesc, C, alpha_s, beta, gamma_bg_12 = theta
    
        if 0. <= fesc <= 1. and 0.2 < C < 10. \
        and -2.5 <= alpha_s <= -1. \
        and 0 <= gamma_bg_12 <= 10 \
        and -3 < beta < -1:
            return 0.
        else:
            return -np.inf


def prior_transform(utheta, args):
    """
    for dynesty utheta = U(0,1)
    """
    fix_bg_bool = args
    
    if fix_bg_bool is True:
        ufesc, uC, ualpha_s, ubeta = utheta
        fesc    = ufesc                # 0 - 1
        C       = 10.2*uC + 0.2        # 0.2 - 10
        alpha_s = -(1.5*ualpha_s + 1.) # 
        beta    = 2*ubeta - 3.         # -3 - -1
        return fesc, C, alpha_s, beta      
    else:
        ufesc, uC, ualpha_s, ubeta, ugamma_bg_12 = utheta
        fesc    = ufesc                # 0 - 1
        C       = 10.2*uC + 0.2        # 0.2 - 10
        alpha_s = -(1.5*ualpha_s + 1.) #
        beta    = 2*ubeta - 3.         # -3 - -1
        gamma_bg_12 = 20.*ugamma_bg_12 # 0 - 20 e-12
        return fesc, C, alpha_s, beta, gamma_bg_12

    
def lnposterior(theta, vlim, sigma_v, Muv, Muv_err, z, fix_bg):
    """Posterior"""
    if lnprior(theta, fix_bg) == 0:
        return lnlike(theta, vlim, sigma_v, Muv, Muv_err, z, fix_bg)
    else:
        return -np.inf    


def load_dynesty_samples(chain_file, save=False):
    """Load dynesty samples and get samples + log Z"""

    res = pickle.load(open(chain_file, 'rb'))
    
    samples = res.samples  # samples
    weights = np.exp(res.logwt - res.logz[-1])  # normalized weights
    samples = dyfunc.resample_equal(samples, weights) # Resample weighted samples.
    
    if save:
        np.save(chain_file.replace('.pickle','_samples'), samples)
        
    return samples, res.logz[-1]