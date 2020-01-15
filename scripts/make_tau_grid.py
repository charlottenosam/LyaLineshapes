# =====================================================================
# make_tau_grid.py
#
# Make tau_tot for grid of R_HII, xHI
#
# HISTORY:
#   Started: 2019-11-14 C Mason (CfA)
#
# Takes ~40 mins on hobnob
# 
# python make_tau_grid.py &
# python make_tau_grid.py --r_slope 0. &
# python make_tau_grid.py --z_s 8. &
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

# =====================================================================
import argparse            # argument managing

# ==============================================================================
# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
# ---- optional arguments ----
parser.add_argument("--r_slope", type=float, help="xHI(r) power law slope [default = 2]")
parser.add_argument("--z_s", type=float, help="redshift of source [default = 7]")
parser.add_argument("--z_min", type=float, help="minimum redshift of integral [default = 6]")
# ---- flags ------
# parser.add_argument("--noclobber", action="store_true", help="Don't make new LUTs")
args = parser.parse_args()

# =====================================================================
print('###################################################')
print('#   Making tau grid (R_HII, xHI)                  #')
print('###################################################')

r_slope = 2.
if args.r_slope is not None:
    r_slope = args.r_slope
print(' - Using rslope: xHI ~ r^%.0f' % r_slope)

z_s = 7.
if args.z_s:
    z_s = args.z_s
print(' - Source redshift: z_s = %.1f' % z_s)

z_min = 6.
if args.z_min:
    z_min = args.z_min
print(' - Minimum redshift: z_min = %.1f' % z_min)

# =====================================================================

# R_ion_tab  = np.arange(0.1, 10, 0.2) * u.Mpc
R_ion_tab  = np.arange(0.01, 5.1, 0.1) * u.Mpc # proper
xHI_01_tab = 10**np.arange(-9, 0.5, 0.5) #np.logspace(-9, 0, 10)

# =====================================================================

max_iter = R_ion_tab.size * xHI_01_tab.size

print('###################################################')
print('Calculating tau')

start = time.time()
tau_total_dict = {}
for i, (R_ion, xHI_01) in enumerate(it.product(R_ion_tab, xHI_01_tab)):
    
    R_ion_key  = np.round(R_ion.value, 2)
    xHI_01_key = np.round(np.log10(xHI_01),2)
    
    if (i+1) % 20 == 0:
        print('%.0f %%\t-- ' % (100*float(i+1)/max_iter),R_ion_key, xHI_01_key)

    tau_tab = bubbles.make_tau_grid(R_ion=R_ion, xHI_01=xHI_01, r_slope=r_slope, z_s=z_s, z_min=z_min)
    tau_HII, tau_IGM, tau_total = tau_tab
    
    tau_total_dict[(R_ion_key, xHI_01_key)] = tau_total
    
    del tau_HII, tau_IGM, tau_total
    
end = time.time()
runtime = end - start
print('###################################################')
print('Took %.1f minutes' % (runtime/60.))

# Save
fname_dict = '../data/tau_total_dict_RHII_xHI_grid_zs=%.1f_zmin=%.1f_r=%.0f.pickle' % (z_s, z_min, r_slope)
with open(fname_dict, 'wb') as fp:
    pickle.dump(tau_total_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
print('\nSaved to %s' % fname_dict)