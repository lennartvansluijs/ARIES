#!/usr/bin/env python
# coding: utf-8

# # Post-processing: automatic wavelength calibration (WASP-33)
# ---
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scipy import polyval
from itertools import product

from astropy.io import fits
import sys
from settings import ARIES_BASE_DIR, DATA_BASE_DIR, CONFIG_BASE_DIR
from settings import WAVCAL_CORR
sys.path.append(ARIES_BASE_DIR)

from aries.cleanspec import clip_mask
from aries.ipfit import correct_continuum
from aries.cleanspec import SpectralCube, SpectralOrder
from aries.preprocessing import plot_image
from aries.constants import TABLEAU20
from aries.constants import ARIES_NORDERS
from aries.cleanspec import clip_mask
from aries.wavcal import WavCalGUI
from IPython.display import display, Math
import corner

# ---
# Program header
# ---


sf, nc = 20, 30
print('\n'+'-'*(2*sf+nc))
print('-'*sf + '      MEASURE autowavcal     ' + '-'*sf)
print('-'*sf + '                             ' + '-'*sf)
print('-'*sf + '     by Lennart van Sluijs   ' + '-'*sf)
print('-'*(2*sf+nc))


# ---
# Parse input parameters
# ---


parser = argparse.ArgumentParser()
parser.add_argument('-targetname')
parser.add_argument('-obsdate')

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate

OBSERVATION_BASE_DIR = os.path.abspath(DATA_BASE_DIR + f'/{targetname}/{obsdate}')

# ## Load telluric model

# ATRAN telluric model: https://atran.arc.nasa.gov/cgi-bin/atran/atran.cgi

# In[4]:


# load telluric model 1.3-2.6 micron
dirin = os.path.abspath(ARIES_BASE_DIR+'/models/telluric')
fname = os.path.join(dirin, 'atran_telluric_model_1_3-2_6_micron_pwv_5mm.txt')
fpath = os.path.join(dirin, fname)
data = np.loadtxt(fpath)
telluric_wav, telluric_spec = data.T


# ## Load Aligned Spectral Time Series

# First let's load the spectral time series for all spectral orders.

# In[6]:


dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/alignment_with_stretching')
spectralorders = []
orders = np.sort(np.array([int(o[6:]) for o in os.listdir(dirin)]))

#orders = [24]
for order in orders:
    fpath = os.path.abspath(dirin+f'/order_{order}/{targetname}_order_aligned.fits')
    data = fits.getdata(fpath)

    fpath = os.path.abspath(dirin+f'/order_{order}/{targetname}_order_mask.fits')
    mask = fits.getdata(fpath)

    spectralorders.append(SpectralOrder(norder=order, data=data, mask=mask, target=targetname))


# ## Automatic Wavelength calibration
import corner

from astropy.io import fits
import sys
ARIES_BASE_DIR = '../..'
sys.path.append(ARIES_BASE_DIR)

from aries.cleanspec import SpectralCube
from aries.cleanspec import SpectralOrder
from aries.preprocessing import plot_image
from aries.constants import TABLEAU20
from aries.wavcal import get_result_mcmc, plot_corner, plot_walkers
from aries.wavcal import AutoWavCalMCMCSampler


ALIGNMENT_MASK_DX = 15 # width around lines used for alignment mask
ALIGNMENT_TEMPLATE_MODE = 'median'
ALIGNMENT_OSR = 2 # oversampling used to interpolate between integer shifts
ALIGNMENT_SHIFT_MAX = 5 # pixel maximal shift


# ## Wavelength calibration based on Brogi et al. 2016
do_continuumcorr = True
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/autowavcal')
if not os.path.exists(dirout):
    os.mkdir(dirout)
os.environ["OMP_NUM_THREADS"] = "1"
for so in spectralorders:

    # Clip data
    data, clip = clip_mask(so.data, so.mask, return_clip=True)
    nframes, nx = data.shape
    data_pxl = so.wavsolution[0][~clip]

    # Get continuum corrected average spectrum
    data_spec = np.median(data, axis=0)
    if do_continuumcorr:
        data_spec = correct_continuum(data_pxl, data_spec, do_plot=False, polydeg=3)

    # Get prior
    configfile = CONFIG_BASE_DIR + f'/wavcal/wasp33_wavcal_order_{so.norder}_wavsolution_coefs.npy'
    coefs_wavsolution = np.load(configfile)
    ncoefs = len(coefs_wavsolution)
    wavcal_corr = np.zeros(ncoefs)
    wavcal_corr[-1] = WAVCAL_CORR[-1]
    theta_prior = tuple(np.array([c for c in coefs_wavsolution]) + wavcal_corr) # correction term is added based on a known offset for this night of observations
    # with respect to the WASP-33 b data set

    #     mcmc_params = {
    #         "theta_prior" : theta_prior,
    #         "nwalkers" : 8,
    #         "nsteps" : int(1e3),
    #         "burn_in" : int(1e2),
    #         "use_pool" : False,
    #         "dtheta" : 5e-3
    #     }
    mcmc_params = {
        "theta_prior" : theta_prior,
        "nwalkers" : 8,
        "nsteps" : int(1e3),
        "burn_in" : int(1e2),
        "use_pool" : True,
        "dtheta" : 5e-3
    }
    #spectralorders_list = [so for i, so in enumerate(spectralorders) if not so.norder in np.arange(23)]

    dirout_autowavcal = os.path.abspath(OBSERVATION_BASE_DIR+f'/processed/spectra/autowavcal/')
    if not os.path.exists(dirout_autowavcal):
        os.makedirs(dirout_autowavcal)


    dirout_order = os.path.abspath(os.path.join(dirout_autowavcal, f'order_{so.norder}'))
    if not os.path.exists(dirout_order):
        os.makedirs(dirout_order)
    print(f'Autowavcal for spectral order {so.norder}')

    autowavcalsampler = AutoWavCalMCMCSampler(
        data_pxl=data_pxl,
        data_spec=data_spec,
        telluric_wav=telluric_wav,
        telluric_spec=telluric_spec,
        wavcoefs_prior=theta_prior
    )
    result = autowavcalsampler.run(
        theta_prior=mcmc_params['theta_prior'],
        nwalkers=mcmc_params['nwalkers'],
        nsteps=mcmc_params['nsteps'],
        use_pool=mcmc_params['use_pool'],
        dtheta=mcmc_params['dtheta']
    )

    if result is None:
        print(f"MCMC unable to find a wavelength solution for order: {so.norder}.")
    else:
        samples = result.get_chain()
        fig, axes = plot_walkers(samples)
        axes[0].set_title("Walkers autowavcal polynomical coefs", size=15)
        plt.savefig(os.path.join(dirout_order, f'autowavcal_walkers_order_{so.norder}.png'))
        plt.close()

        flat_samples = result.get_chain(discard=mcmc_params['burn_in'], flat=True)
        fig, axes = plot_corner(flat_samples)
        plt.suptitle('Corner plot autowavcal polynomial coefs', size=15)
        params_result = get_result_mcmc(flat_samples, verbose=False)
        plt.savefig(os.path.join(dirout_order, f'autowavcal_marginals_order_{so.norder}.png'))
        plt.close()

        # Finally calculate the wavsolution using the best wavcoefs
        wavcoefs_best = [params_result[f'c{i}'][0] for i in range(len(theta_prior))][::-1]
        wavcoefs_best_lowlim = [params_result[f'c{i}'][1] for i in range(len(theta_prior))][::-1]
        wavcoefs_best_uplim = [params_result[f'c{i}'][2] for i in range(len(theta_prior))][::-1]

        data_wavsolution = polyval(wavcoefs_best, data_pxl)
        data_wavsolution_prior = polyval(theta_prior, data_pxl)
        plt.plot(data_pxl, data_wavsolution,color='b', label='autowavcal solution')
        plt.plot(data_pxl, data_wavsolution_prior, color='k', ls='--', label='prior')
        plt.xlabel('pixel')
        plt.ylabel('wavelength (micron)')
        plt.savefig(os.path.join(dirout_order, f'autowavcal_wavsolution_order_{so.norder}.png'))
        plt.close()

        telluric_spec_interp = np.interp(x=data_wavsolution, xp=telluric_wav, fp=telluric_spec)
        telluric_spec_interp_prior = np.interp(x=data_wavsolution_prior, xp=telluric_wav, fp=telluric_spec)
        plt.plot(data_wavsolution, data_spec, label='data')
        plt.plot(data_wavsolution, telluric_spec_interp, label='wavsolution')
        plt.plot(data_wavsolution, telluric_spec_interp_prior, label='prior')
        plt.xlabel('wavelength (micron)')
        plt.ylabel('flux')
        plt.legend()
        plt.savefig(os.path.join(dirout_order, f'autowavcal_telluric_fit_order_{so.norder}.png'))
        plt.close()

        np.savetxt(os.path.join(dirout_autowavcal, f'{so.target}_wavcal_order_{so.norder}_wavsolution.txt'),
                           np.c_[data_pxl, data_wavsolution],
                           header = 'order={}: data_wav [pixel], data_wavsolution [micron]'.format(so.norder))
        np.save(os.path.join(dirout_autowavcal, f'{so.target}_wavcal_order_{so.norder}_wavsolution_coefs.npy'),
                             wavcoefs_best)
