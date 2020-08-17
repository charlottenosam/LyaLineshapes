# =====================================================================
# bluepeak_inference_dynesty.py
#
# Run blue peak inference with dynesty
#
# HISTORY:
#   Started: 2020-04-14 C Mason (CfA)
# 
#
# %run bluepeak_inference_dynesty 'res_gamma10_fixNion_fixverr' --maxiter 100000
# %run bluepeak_inference_dynesty 'res_nobg_gamma10_fixNion_fixverr' --fix_bg --maxiter 100000
# %run bluepeak_inference_dynesty 'res_gamma10_fixNion_lnD_fixverr' --maxiter 100000
# %run bluepeak_inference_dynesty 'res_nobg_gamma10_fixNion_lnD_fixverr' --fix_bg --maxiter 100000

#
# =====================================================================
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import math
import scipy
import os, glob, sys
import pickle
import time
import itertools as it

from astropy.cosmology import Planck15
from astropy import units as u
from astropy import constants as const

sys.path.append('../')
import bubbles

# Dynesty imports
import pickle
import dynesty
from dynesty import plotting as dyplot
from dynesty import DynamicNestedSampler
from dynesty import utils as dyfunc

from multiprocessing import Pool
import ipyparallel as ipp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# To use multiprocessing run following command:
# > ipcluster start -n 7 &
rc = ipp.Client()
nprocs = len(rc.ids)
print(rc.ids)
dview = rc[:]
dview.use_dill();

# =====================================================================
import argparse            # argument managing

# ==============================================================================
# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("file_name", type=str, help="File name, saved in ../chains/")
# ---- optional arguments ----
parser.add_argument("--maxiter", type=int, help="Max iterations [default = 50000]")
# ---- flags ------
parser.add_argument("--fix_bg", action="store_true", help="fix ionizing background [default = False]")
parser.add_argument("--noplots", action="store_true", help="don't make plots [default = False]")
args = parser.parse_args()
# =====================================================================

print('###################################################')
print('#   Running dynesty inference                     #')
print('###################################################\n')

# =====================================================================
# Setup args

labels = [r'$f_\mathrm{esc}$', r'$C_\mathrm{HII}$', r'$\alpha$', r'$\beta$', r'$\Gamma_\mathrm{bg} [10^{-12} \mathrm{s}^{-1}]$']
labels = [r'$f_\mathrm{esc}$', r'$C_\mathrm{HII}$', r'$\ln{\Delta}$', r'$\alpha$', r'$\beta$', r'$\Gamma_\mathrm{bg} [10^{-12} \mathrm{s}^{-1}]$']
fix_bg = False
ndim   = len(labels)
if args.fix_bg:
    fix_bg = True
    ndim   = ndim-1
    labels = labels[:ndim]
    print(' - Fixing ionizing background')
else: 
    print(' - Free ionizing background')

maxiter = 50000
if args.maxiter:
    maxiter = args.maxiter
print(' - Max iter:', maxiter)

plot = True
if args.noplots:
    plot = False

chain_file = "../chains/%s_N=%i.pickle" % (args.file_name, maxiter)
print(' - Saving to',chain_file)

# =====================================================================

# vlim, sigma_v, Muv, Muv_err, z, = 250.*u.km/u.s, 60.*u.km/u.s, -21.6, 0.3, 6.6
vlim, sigma_v, Muv, Muv_err, z, = 250.*u.km/u.s, 10.*u.km/u.s, -21.6, 0.3, 6.6

infer = bubbles.blue_peak_inference(vlim, sigma_v, Muv, Muv_err, z, 
                                    fix_bg=fix_bg, 
                                    gamma_bg_bounds=[-2., 2.], log_gamma_bg=True,
                                    C_bounds=[1., 10.], log_C=False)
# =====================================================================

# if __name__ == '__main__':
   
# sample from the target distribution
t0 = time.time()
npool = 7
with ProcessPoolExecutor(max_workers=npool) as executor:
    
    sampler = DynamicNestedSampler(
                            infer.lnlike, infer.prior_transform, ndim, 
                            pool=executor, queue_size=npool,
                            bound='multi', sample='unif')
    
    sampler.run_nested(
    					dlogz_init=0.05, nlive_init=1000, 
                        nlive_batch=100,
    					maxiter=maxiter, 
    					use_stop=False, 
    					# wt_kwargs={'pfrac': 1.0}
    					)
    
    res = sampler.results        
    pickle.dump(res, open(chain_file,"wb"))

t_run = (time.time()-t0)/60.
print('\n============================================')
print("Sampling took {0:.10f} mins".format(t_run))
print('============================================')

# =====================================================================
# Plots
if plot:
	rfig, raxes = dyplot.runplot(res, span=[0.0, (0., 1.1), 0.0, (0., 1.05*np.exp(np.nanmax(res.logz)))])
	plt.savefig(chain_file.replace('.pickle','_runplot.png'))

	tfig, taxes = dyplot.traceplot(res, labels=labels)
	plt.savefig(chain_file.replace('.pickle','_traceplot.png'))

	cfig, caxes = dyplot.cornerplot(res, labels=labels, show_titles=True)
	plt.savefig(chain_file.replace('.pickle','_cornerplot.png'))

del res
