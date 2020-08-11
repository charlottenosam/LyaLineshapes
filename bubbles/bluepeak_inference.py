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
import time

from astropy.cosmology import Planck15
from astropy import units as u
from astropy import constants as const

import emcee, corner
from multiprocessing import Pool

# Dynesty imports
import pickle
import dynesty
from dynesty import utils as dyfunc

import bubbles

# =====================================================================
np.random.seed(42)

# =====================================================================

class blue_peak_inference(object):
    """
    Blue peak inference routines
    """

    def __init__(self, vlim, sigma_v, Muv, Muv_err, z, fix_bg,
                    fesc_bounds=[0.,1.], C_bounds=[0.1, 10.],
                    sigma0=1.,
                    alpha_bounds=[-2.5,-1.], beta_bounds=[-3,-1],
                    gamma_bg_bounds=[0.,20.], gamma_bg_rescale=1e-12,
                    log_C=False, log_gamma_bg=False):
        """
        Default gamma_bg is * 1e-12 s^-1, if log, it's not
        """
        # input parameters
        self.vlim    = vlim
        self.sigma_v = sigma_v
        self.Muv     = Muv
        self.Muv_err = Muv_err
        self.z       = z
        self.fix_bg  = fix_bg
        self.gamma_bg_rescale = gamma_bg_rescale

        if self.fix_bg:
            self.ndim = 5
        else:
            self.ndim = 6

        # params bounds
        self.prior_bounds = {'fesc':fesc_bounds, 'C':C_bounds,
                            'alpha_s':alpha_bounds, 'beta':beta_bounds,
                            'gamma_bg':gamma_bg_bounds, 'sigma0':sigma0}

        self.R_obs     = np.abs(self.vlim / Planck15.H(self.z)).to(u.Mpc)
        self.R_obs_err = np.abs(self.sigma_v / Planck15.H(self.z)).to(u.Mpc)

        # labels
        self.labels = [r'$f_\mathrm{esc}$', r'$C_\mathrm{HII}$', r'$\Delta$', 
                        r'$\alpha$', r'$\beta$', 
                        r'$\Gamma_\mathrm{bg} [10^{-12} \mathrm{s}^{-1}]$']

        # Use log (gamma s^-1)?
        self.log_gamma_bg = log_gamma_bg
        if self.log_gamma_bg:
            self.labels[-1] = r'$\log_{10} \Gamma_\mathrm{bg} [\mathrm{s}^{-1}]$'

        # Use log C?
        self.log_C = log_C
        if self.log_C:
            self.prior_bounds['C'] = [np.log10(C_bounds[0]), np.log10(C_bounds[1])]
            self.labels[1] = r'$\log_{10} C_\mathrm{HII}$'

        print('Prior bounds:',self.prior_bounds)

        return

# ------------------------------------------------------------

    def emcee_setup(self, nwalkers=100):
        """
        Set initial position of chains
        """

        nll = lambda *args: -self.lnlike(*args)

        # fesc, C, lnDelta, alpha_s, beta = theta
        initial = np.array([1., 1., 0., -2., -2., 1.])
        if self.fix_bg is True:
            initial = initial[:-1]

        pos = initial + 1e-1 * np.random.randn(nwalkers, len(initial))

        return pos


    def emcee_run(self, pos, Nsteps=2000):
        """
        Run emcee
        """
        nwalkers, ndim = pos.shape
        print('%i walkers, %i dimensions' % (nwalkers, ndim))
        assert ndim == self.ndim, 'Wrong number of dimensions!'

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnposterior, 
                                            pool=pool)
            start = time.time()
            sampler.run_mcmc(pos, Nsteps, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        # Autocorrelation time
        tau = sampler.get_autocorr_time(quiet=True)
        print(tau)

        return sampler


# ------------------------------------------------------------

    def lnposterior(self, theta):
        """Posterior"""
        if np.isfinite(self.lnprior(theta)):
            return self.lnlike(theta)+self.lnprior(theta)
        else:
            return -np.inf    


    def lnlike(self, theta):
        """
        Likelihood for observing vlim+sigma_v given theta
        
        Likelihood is that R_alpha > vlim/H(z) --> erfc
        
        """
        
        if self.fix_bg is True:
            fesc, C, lnDelta, alpha_s, beta = theta
            J_bg = 0.
        
        else:
            fesc, C, lnDelta, alpha_s, beta, gamma_bg = theta    
            if self.log_gamma_bg:
                # NOT /10^-12
                gamma_bg = 10**gamma_bg
            else:
                gamma_bg = gamma_bg * self.gamma_bg_rescale
            J_bg = gamma_bg / bubbles.Gamma12(self.z)

        Delta = np.exp(lnDelta)

        if self.log_C:
            C = 10**C
        
        # Ionizing photon flux
        Muv_draw = np.random.normal(loc=self.Muv, scale=self.Muv_err)
        Ndot_ion = bubbles.Muv_to_Nion(Muv_draw, self.z, alpha_s=alpha_s, beta=beta)
        
        # Model proximity zone radius
        R_mod = bubbles.R_optically_thin(self.z, Ndot_ion=Ndot_ion, alpha_s=alpha_s, 
                                         fesc=fesc, C=C, Delta=Delta,
                                         reccase='B', T=1e4*u.K, tau_lim=2.3, 
                                         J_bg=J_bg)
        
        if np.isnan(R_mod.value):
            R_mod = np.inf * u.Mpc
        
        likelihood = 0.5 * scipy.special.erfc((self.R_obs - R_mod) / np.sqrt(2.) / self.R_obs_err)

        return np.log(likelihood)


# ------------------------------------------------------------

    def lnprior(self, theta):
        """For emcee

        uniform priors on everything
        lognormal prior on delta
        """
        if self.fix_bg:
            # fesc, C, alpha_s, beta = theta
            fesc, C, lnDelta, alpha_s, beta = theta

            p_lnDelta = -0.5*(np.log(2*np.pi*self.prior_bounds['sigma0']) + \
                        ((lnDelta + 0.5*self.prior_bounds['sigma0']**2.)/self.prior_bounds['sigma0'])**2.)

            if self.prior_bounds['fesc'][0] <= fesc <= self.prior_bounds['fesc'][1] \
            and self.prior_bounds['C'][0] <= C <= self.prior_bounds['C'][1] \
            and self.prior_bounds['alpha_s'][0] <= alpha_s <= self.prior_bounds['alpha_s'][1] \
            and self.prior_bounds['beta'][0] <= beta <= self.prior_bounds['beta'][1]:
                # return 0.
                return p_lnDelta
            else:
                return -np.inf
            
        else:
            fesc, C, lnDelta, alpha_s, beta, gamma_bg = theta

            p_lnDelta = -0.5*(np.log(2*np.pi*self.prior_bounds['sigma0']) + \
                        ((lnDelta + 0.5*self.prior_bounds['sigma0']**2.)/self.prior_bounds['sigma0'])**2.)

            if self.prior_bounds['fesc'][0] <= fesc <= self.prior_bounds['fesc'][1] \
            and self.prior_bounds['C'][0] < C < self.prior_bounds['C'][1] \
            and self.prior_bounds['alpha_s'][0] <= alpha_s <= self.prior_bounds['alpha_s'][1] \
            and self.prior_bounds['beta'][0] < beta < self.prior_bounds['beta'][1] \
            and self.prior_bounds['gamma_bg'][0] < gamma_bg < self.prior_bounds['gamma_bg'][1]:
                return p_lnDelta
            else:
                return -np.inf


    def prior_transform(self, utheta):
        """
        for dynesty utheta = U(0,1)
        """
        fix_bg_bool = args
        
        if fix_bg_bool is True:
            ufesc, uC, ulnDelta, ualpha_s, ubeta = utheta
            fesc    = (self.prior_bounds['fesc'][1] - self.prior_bounds['fesc'][0])*ufesc + self.prior_bounds['fesc'][0]
            C       = (self.prior_bounds['C'][1] - self.prior_bounds['C'][0])*uC + self.prior_bounds['C'][0]
            lnDelta = scipy.stats.norm.pdf(ulnDelta, loc=-0.5*self.prior_bounds['sigma0']**2., scale=self.prior_bounds['sigma0'])
            alpha_s = (self.prior_bounds['alpha_s'][1] - self.prior_bounds['alpha_s'][0])*ualpha_s + self.prior_bounds['alpha_s'][0]
            beta    = (self.prior_bounds['beta'][1] - self.prior_bounds['beta'][0])*ubeta + self.prior_bounds['beta'][0]
            return fesc, C, lnDelta, alpha_s, beta      
        else:
            ufesc, uC, ulnDelta, ualpha_s, ubeta, ugamma_bg = utheta
            fesc     = (self.prior_bounds['fesc'][1] - self.prior_bounds['fesc'][0])*ufesc + self.prior_bounds['fesc'][0]
            C        = (self.prior_bounds['C'][1] - self.prior_bounds['C'][0])*uC + self.prior_bounds['C'][0]
            lnDelta  = scipy.stats.norm.pdf(ulnDelta, loc=-0.5*self.prior_bounds['sigma0']**2., scale=self.prior_bounds['sigma0'])
            alpha_s  = (self.prior_bounds['alpha_s'][1] - self.prior_bounds['alpha_s'][0])*ualpha_s + self.prior_bounds['alpha_s'][0]
            beta     = (self.prior_bounds['beta'][1] - self.prior_bounds['beta'][0])*ubeta + self.prior_bounds['beta'][0]
            gamma_bg = (self.prior_bounds['gamma_bg'][1] - self.prior_bounds['gamma_bg'][0])*ugamma_bg + self.prior_bounds['gamma_bg'][0]
            return fesc, C, lnDelta, alpha_s, beta, gamma_bg

        
# ------------------------------------------------------------

    def plot_emcee_chains(self, sampler):
        """Plot chains"""
        samples = sampler.get_chain()
        
        fig, axes = plt.subplots(self.ndim, figsize=(8, self.ndim*2), sharex=True)

        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.1, lw=1)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

        return


    def plot_emcee_corner(self, sampler, discard=300, thin=50, smooth=1, plotname=None):
        """
        Plot corner

        Lines show 1,2,3 sigma regions

        Autocorrelation time suggests that only about X steps 
        are needed for the chain to “forget” where it started.

        It’s not unreasonable to throw away a few times this number of steps as “burn-in”.
        Let’s discard the initial 2X steps, thin by about half the autocorrelation time (X/2)
        and flatten the chain so that we have a flat list of samples:
        """

        flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        flat_samples[:,2] = np.exp(flat_samples[:,2]) # lnDelta --> Delta
        print('Reshaped to:',flat_samples.shape)

        fig, ax = plt.subplots(self.ndim, self.ndim, figsize=(self.ndim+1.4, self.ndim+1.5), dpi=150)
        corner.corner(flat_samples, fig=fig,
                        labels=self.labels, smooth=smooth, 
                        color='Teal', use_math_text=True,
                        plot_datapoints=False, plot_density=False, 
                        fill_contours=True, hist_kwargs={'lw':2},
                        levels = 1.0 - np.exp(-0.5 * np.array([1,2,3]) ** 2),
                        show_titles=True, 
                        label_kwargs={"fontsize": 16})
        
        if plotname is not None:
            plt.savefig(plotname, bbox_inches='tight')

        return


    def load_dynesty_samples(self, chain_file, save=False):
        """Load dynesty samples and get samples + log Z"""

        res = pickle.load(open(chain_file, 'rb'))
        
        samples = res.samples  # samples
        weights = np.exp(res.logwt - res.logz[-1])  # normalized weights
        samples = dyfunc.resample_equal(samples, weights) # Resample weighted samples.
        
        if save:
            np.save(chain_file.replace('.pickle','_samples'), samples)
            
        return samples, res.logz[-1]