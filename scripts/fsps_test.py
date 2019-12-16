# =====================================================================
# fsps_Nion.py
#
# Make Nion from SPS
#
# HISTORY:
#   Started: 2019-12-12 C Mason (CfA)
#
#
# Grid of tage, SFH=4 ()

# =====================================================================
import fsps
import matplotlib.pylab as plt
import numpy as np
from astropy.cosmology import Planck15
from astropy import units as u
from astropy import constants as const

sys.path.append('../')
import bubbles
import itertools as it

import fsps
sp = fsps.StellarPopulation(zcontinuous=1., imf_type=0)
# zred=8., logzsol=-1.7, )  
sp.params['zred'] = 0. # redshift sets IGM transmission
sp.params['sfh'] = 4. # delayed tau
sp.params['const'] = 1. # constant SFH on top of general SFH

# parameters to vary
tage_grid = np.array([1e-3, 1e-2, 1e-1])
logzsol_grid = np.linspace(-2, 0, 2)
# mstar_grid = np.logspace(6, 10, 2)
sfr_grid = np.logspace(-3, 3)

# How rare are objects with these ionizing fluxes? Sandro's model?

def Nion(wave, spec, wave_lim=912.):
    """
    spec [L_sun/Hz]
    wave --> frequency
    """
    nu = (const.c / wave/u.Angstrom).to(u.Hz)
    spec_ergsHz = spec*3.846e33*u.erg/u.s/u.Hz

    integral_limits = np.where((wave < wave_lim) & (spec > 1e-30))

    nu_int = nu[integral_limits][::-1]
    spec_ergsHz_int = spec_ergsHz[integral_limits][::-1]
    
    Nion = np.trapz(spec_ergsHz_int/const.h/nu_int, nu_int).to(1/u.s)
    return Nion


# Formed 1 solar mass
# Structured numpy array!!!
# sp.stellar_mass is mass at tage
dtype = np.dtype([('Nion',np.float), ('SFR',np.float), 
        ('age',np.float), ('logzsol',np.float), ('mstar',np.float)])

results_list = []
for age, logz, sfr_grid in it.product(tage_grid, logzsol_grid, sfr_grid):

    sp.params['logzsol'] = logz
    wave, spec = sp.get_spectrum(tage=age)

    mstar - 
    # Spectrum for this stellar mass in Lsun/Hz
    spec *= mstar/sp.stellar_mass # total mass formed

    results = np.zeros(1, dtype=dtype)
    results['age'] = age
    results['logzsol'] = logz
    results['mstar'] = mstar

    results['Nion'] = Nion(wave, spec).value

    results['SFR'] = mstar/sp.stellar_mass/age 

    results_list.append(results)

# results_list = np.array(results_list)
results_all = np.concatenate(results_list)


plt.axvline(912., c='k', ls='dotted', lw=1)                          
for age in [1e-3, 1e-2, 1e-1, 1]: 
    plt.loglog(*sp.get_spectrum(tage=age, peraa=True), label='%.0f' % np.log10(age*1e9)) 

# L_sun/Hz                                                                   

plt.ylim(1e-15, 10)                                                  
plt.xlim(1e2, 1e3)                                                   

# Print stellar libraries
sp.libraries                                                         
sp.isoc_library                                                      
