#!/usr/bin/env python
# coding: utf-8

# # WASP-33b run GCM models

# In[28]:

import argparse
import time
import csv
import os
import pickle
import sys
import warnings

from astropy.time import Time
from astropy.io import fits
import astropy.units as u

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

ARIES_BASE_DIR = os.path.abspath('../..')
sys.path.append(ARIES_BASE_DIR)
sys.path.append(os.path.abspath(ARIES_BASE_DIR)+'/lib')

from aries.cleanspec import SpectralOrder, TemplateOrder
from aries.cleanspec import planck
from aries.constants import TABLEAU20
from aries.constants import ARIES_SPECTRAL_RESOLUTION
from aries.constants import ARIES_NORDERS
from aries.crosscorrelate import get_rvplanet
from aries.crosscorrelate import calc_orbital_phase
from aries.crosscorrelate import estimate_kp
from aries.crosscorrelate import calc_ccmatrix, calc_ccmatrix_2d, plot_detection_matrix, plot_ccmatrices
from aries.crosscorrelate import calc_detection_matrices2
from aries.crosscorrelate import get_planet_params
from aries.crosscorrelate import shift_to_new_restframe
from aries.crosscorrelate import plot_ccmatrix
from aries.crosscorrelate import ttest_on_trails
from aries.ipfit import gaussian_convolution_kernel
from aries.preprocessing import get_keyword_from_headers_in_dir, get_fits_fnames
from aries.systemparams import systems, calc_transit_duration, targetname_to_key
from aries.utils import phase_filter
from aries.fastloglike import BrogiLineBayesianFramework_2DTemplate, BrogiLineBayesianFramework_1DTemplate


# In[29]:

# start timer
ts = time.time()


# In[30]:

# ---
# Parse input parameters
# ---


parser = argparse.ArgumentParser()
parser.add_argument('-model_fname')
parser.add_argument('-dirin_data')
parser.add_argument('-dirout_significances')
parser.add_argument('-dirname_models')
parser.add_argument('-apply_hpf_to_model', type=bool)
parser.add_argument('-obsdates', type=str)
parser.add_argument('-targetname', type=str)
parser.add_argument('-phase_filter_mode', type=str)
parser.add_argument('-orders', type=str)


args = parser.parse_args()
model_fname = args.model_fname
dirname_models = args.dirname_models
apply_hpf_to_model = args.apply_hpf_to_model
obsdates = args.obsdates.split(' ')
targetname = args.targetname
phase_filter_mode = args.phase_filter_mode
orders = np.array(args.orders.split(' '), dtype=int)


continuum_subtracted = False


dirin_models = os.path.abspath(ARIES_BASE_DIR + f"models/{targetname}/{dirname_models}")
#model_fname = "phase_dependent.txt"
fpath_model = os.path.join(dirin_models, model_fname)
NO_TXT_EXTENSION = slice(0,-4,1)

systemparams = systems[targetname_to_key(targetname)]

if len(obsdates) == 1:
    dirout_significances = os.path.abspath(ARIES_BASE_DIR +f"/data/{targetname}/{obsdates[0]}/processed/injected/{args.dirout_significances}")
    print('Using a single night.')
elif len(obsdates) > 1:
    dirout_significances = os.path.abspath(ARIES_BASE_DIR +f"/data/{targetname}/all/processed/injected/{args.dirout_significances}")
    print('Combing all nights.')
else:
    raise ValueError('Invalid entry for obsdates.')
if not os.path.exists(dirout_significances):
    os.makedirs(dirout_significances)

run_crosscorr = True

use_crosscorr_weighting_scheme = False
TTEST_TRAIL_WIDTH = 3
TTEST_OUT_OF_TRAIL_RADIUS = 5

nx, ny = (31, 31)
dvsys_all = np.linspace(-100e3, 100e3, nx)
dkp_all = np.linspace(-150e3, 150e3, ny)

RV_MIN = -500e3 # m/s
RV_MAX = 500e3 # m/s
DELTA_RV = 5e3 
nshifts = int((RV_MAX-RV_MIN)/DELTA_RV + 1)
rv_sample = np.linspace(RV_MIN, RV_MAX, nshifts)

result = {'snr':{}, 'ttest':{}, 'logL_grid':{}, 'logL_multinest':{}} # allocate memory to save results

# In[32]:


from imp import reload
import aries
reload(aries.cleanspec)
from aries.cleanspec import TemplateOrder

# Load all parameters up to point of running loglike.
spectralorders_all = []
templateorders_all = []

for obsdate in obsdates:
    #  Define observing date specific input directories
    OBSERVATION_BASE_DIR = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}')
    dirin_residuals = os.path.abspath(ARIES_BASE_DIR + f"/data/{targetname}/{obsdate}/processed/injected/{args.dirin_data}")
    dirin_meta = os.path.abspath(OBSERVATION_BASE_DIR+'/meta')
    dirin_templates = os.path.abspath(OBSERVATION_BASE_DIR + f'/processed/hrccs_templates/{dirname_models}/' + os.path.basename(fpath_model)[NO_TXT_EXTENSION])
    if apply_hpf_to_model:
        dirin_templates += '_hpf'

    print('\nInput parameters:')
    print(f'\tDate\t: {obsdate}')
    print(f'\tResiduals input directory\t: {dirin_residuals}')
    print(f'\tMeta data input directory\t: {dirin_meta}')
    print(f'\tSignificances output directory\t: {dirout_significances}')
    print(f'\tTemplates input directory\t: {dirin_templates}')
    
    #  Load barycentric corrected time stamps and orbital phase
    print('\n\tLoading observed times, phases and barycentric data...')
    
    fpath_times = os.path.join(dirin_meta, 'science_times.txt')
    times_bjdtbd = np.loadtxt(fpath_times, dtype='str', delimiter=',').T[4].astype('float')
    fpath_vbary = os.path.join(dirin_meta, 'vbarycorr.txt')
    phase, vbary, rvplanet = np.loadtxt(fpath_vbary)
    print('Done.')
    
    
    #  Load residuals
    print('\n\tLoading observed residuals...')
    
    data_all = [fits.getdata(os.path.join(dirin_residuals, f'order_{norder}/{targetname}_residual_{norder}.fits')) for norder in orders]
    mask_all = [fits.getdata(os.path.join(dirin_residuals, f'order_{norder}/{targetname}_mask_{norder}.fits')) for norder in orders]
    wavsolution_all = [np.loadtxt(os.path.join(dirin_residuals, f'order_{norder}/{targetname}_wavsolution_{norder}.txt')) for norder in orders]

    spectralorders = []
    for (norder, data, mask, wavsolution) in zip(orders, data_all, mask_all, wavsolution_all):
        so = SpectralOrder(
            data=data,
            mask=mask,
            norder=norder,
            wavsolution=wavsolution,
            target=targetname,
            phase=phase,
            time=times_bjdtbd
        )
        so.vbary = vbary
        so.obsdate = obsdate
        spectralorders.append(so)
    print('Done.')

    # Load templates
    print('\n\tLoading template orders...')
    templateorders = [TemplateOrder.load(f=os.path.join(dirin_templates, f'{targetname}_template_order_{norder}')) for norder in orders]
    print('Done.')
    
    spectralorders_all.append(spectralorders)
    templateorders_all.append(templateorders)

# Combine all observing nights
spectralorders = []
templateorders = []
for i in range(len(obsdates)):
    for j in range(len(orders)):
        so = spectralorders_all[i][j]
        to = templateorders_all[i][j]
        spectralorders.append(so)
        templateorders.append(to)

#  Apply mask to frames outside of selection
for so in spectralorders:
    selected_phases = phase_filter(
        phase_obs=so.phase,
        occultation_phases=systemparams["occultation_phases"],
        mode=phase_filter_mode
    )
    so.mask[~selected_phases,:] = True

#  Remove fully masked orders
spectralorders_s = []
templateorders_s = []
for to, so in zip(templateorders, spectralorders):
    if not np.all(so.mask):
        spectralorders_s.append(so)
        templateorders_s.append(to)
spectralorders = spectralorders_s
templateorders = templateorders_s


# ---
# Cross-correlation method
# ---


if run_crosscorr:
    print('\n\tRunning cross-correlation with model template...')
    dirout = dirout_significances+'/crosscorr/'
    if not os.path.exists(dirout):
        os.mkdir(dirout)
    

    norders = len(spectralorders)
    ccmatrices_allnights = []
    phase_allnights = []
    vbary_allnights = []
    for obsdate in np.unique([so.obsdate for so in spectralorders]):
        ccmatrices = []
        spectralorders_obsdate = [so for so in spectralorders if so.obsdate == obsdate]
        templateorders_obsdate = [to for to in templateorders if to.obsdate == obsdate]
        with tqdm(total= len(spectralorders_obsdate), desc='cross-correlation') as pbar:
            for so, to in zip(spectralorders_obsdate, templateorders_obsdate):
                ccmatrix = calc_ccmatrix_2d(so.data, so.wavsolution[1],
                                     to.wavegrid, to.data,
                                     rv_sample, mask=so.mask, wav_pad=0.15)
                fits.writeto(os.path.join(dirout, '{}_ccmatrix_order_{}_{}.fits'.format(so.target, so.norder, so.obsdate)), ccmatrix, overwrite=True)
                ccmatrices.append(ccmatrix)
                pbar.update()
    
        # Use uniform weights for now
        cc_weights = np.ones(len(spectralorders_obsdate))
        ccmatrix_avg = np.average(ccmatrices, weights=cc_weights, axis=0)
        fits.writeto(os.path.join(dirout, '{}_ccmatrix_avg_{}.fits'.format(so.target, obsdate)), ccmatrix_avg, overwrite=True)

        # Create an overview plot of all cross-correlation results for all orders
        axes = plot_ccmatrices(ccmatrices, ccmatrix_avg, np.zeros(ccmatrix_avg.shape), orders, rv_sample)
        plt.savefig(os.path.join(dirout, f'{targetname}_overview_ccmatrices_{obsdate}.png'), dpi=200)
        plt.close()
        
        ccmatrices_allnights.append(ccmatrix_avg)
        phase_allnights.append(spectralorders_obsdate[0].phase)
        vbary_allnights.append(spectralorders_obsdate[0].vbary)
        
    ccmatrix_allnights = np.concatenate(np.array(ccmatrices_allnights))
    phase_allnights = np.concatenate(np.array(phase_allnights))
    vbary_allnights = np.concatenate(np.array(vbary_allnights))
    ind_sorted = phase_allnights.argsort()
    phase_combined = phase_allnights[ind_sorted]
    vbary_combined = vbary_allnights[ind_sorted]
    ccmatrix_combined = ccmatrix_allnights[ind_sorted, :]
    
    
    # mask phases outside of range
    selected_phases_combined = phase_filter(
        phase_obs=phase_combined,
        occultation_phases=systemparams["occultation_phases"],
        mode=phase_filter_mode
    )

    # Cross correlation matrix for multiple vsys, kp values
    trial_kp, trial_vsys = (dkp_all + systemparams['kp'], dvsys_all + systemparams['vsys'])
    snrmatrix, sigmamatrix = calc_detection_matrices2(ccmatrix=ccmatrix_combined[selected_phases_combined,:],
                                                    dvsys_all=dvsys_all,
                                                    dkp_all=dkp_all,
                                                    phase=phase_combined[selected_phases_combined],
                                                    vbary=vbary_combined[selected_phases_combined],
                                                    vsys=systemparams['vsys'], # used to shift to planet rest frame
                                                    rv_sample=rv_sample,
                                                    kp=systemparams['kp'],
                                                    radius=TTEST_OUT_OF_TRAIL_RADIUS,
                                                    trail_width=TTEST_TRAIL_WIDTH)

    # save T-test and snr matrix
    fname = os.path.join(dirout, 'snrmatrix.fits')
    fits.writeto(fname, snrmatrix, overwrite=True)
    fname = os.path.join(dirout, 'sigmamatrix.fits')
    fits.writeto(fname, sigmamatrix, overwrite=True)
    fname = os.path.join(dirout, 'grid.txt')
    header = f"center: (vsys, kp) = ({systemparams['vsys']}, {systemparams['kp']}) \n"
    "grid: dvsys_all (m/s) | dkp_all (m/s)"
    np.savetxt(fname, np.c_[dvsys_all, dkp_all], header=header, delimiter=',')
    center = (systemparams['vsys'], systemparams['kp'])
    with open(os.path.join(dirout,'center.pickle'), 'wb') as f:
        pickle.dump(center, f)

    axes = plot_detection_matrix(snrmatrix, dkp_all=dkp_all, dvsys_all=dvsys_all,
                                 kp=systemparams['kp'], vsys=systemparams['vsys'],
                                title='Cross-correlation (SNR)', mode='snr')
    plt.savefig(os.path.join(dirout, 'snrmatrix.png'), dpi=200)
    plt.close()

    axes = plot_detection_matrix(sigmamatrix, dkp_all=dkp_all, dvsys_all=dvsys_all,
                                 kp=systemparams['kp'], vsys=systemparams['vsys'],
                                title='Cross-correlation (T-test)', mode='ttest')
    plt.savefig(os.path.join(dirout, 'ttestmatrix.png'), dpi=200)
    plt.close()

    # Cross-correlation matrix for highest SNR.
    vsys_max, kp_max, snr_max = get_planet_params(snrmatrix, trial_kp, trial_vsys)
    result['snr'] = {'significance':snr_max, 'vsys':vsys_max, 'kp':kp_max}
    suptitle = 'Average C-C Matrix (Best fit SNR: SNR={:.2f}, vsys={:.2f} km/s, Kp={:.2f} km/s)'.format(snr_max, vsys_max/1e3, kp_max/1e3)
    rvplanet = -vbary_combined + vsys_max + kp_max*np.sin(phase_combined*2*np.pi) # rv trial, shifted to the planet's rest frame
    ccmatrix_combined_shifted = shift_to_new_restframe(ccmatrix_combined, rv0=rv_sample, rvshift=rvplanet)
    fits.writeto(os.path.join(dirout, 'ccmatrix_shifted_bestfit_snr.fits'), ccmatrix_combined_shifted, overwrite=True)

    axes, cbar = plot_ccmatrix(ccmatrix_combined, ccmatrix_combined_shifted, rv_sample, phase_combined)
    plt.suptitle(suptitle, size=12)
    plt.savefig(os.path.join(dirout, 'ccmatrix_bestfit_snr.png'), dpi=200)
    plt.close()

    _ = ttest_on_trails(ccmatrix_combined_shifted, trail_width=TTEST_TRAIL_WIDTH, radius=TTEST_OUT_OF_TRAIL_RADIUS, plot=True)
    plt.suptitle(suptitle, size=12)
    plt.savefig(os.path.join(dirout, 'ttest_bestfit_snr.png'), dpi=200)
    plt.close()

    # Highest T-Test CC-Matrix.
    vsys_max, kp_max, sigma_max = get_planet_params(sigmamatrix, trial_kp, trial_vsys)
    result['ttest'] = {'significance':sigma_max, 'vsys':vsys_max, 'kp':kp_max}
    suptitle=r'Average C-C Matrix (Best fit T-test: $\sigma$={:.2f}, $vsys$={:.2f} km/s, Kp={:.2f} km/s)'.format(sigma_max, vsys_max/1e3, kp_max/1e3)
    rvplanet = -vbary_combined + vsys_max + kp_max*np.sin(phase_combined*2*np.pi) # rv trial, shifted to the planet's rest frame
    ccmatrix_combined_shifted = shift_to_new_restframe(ccmatrix_combined, rv0=rv_sample, rvshift=rvplanet)
    fits.writeto(os.path.join(dirout, 'ccmatrix_shifted_bestfit_sigma.fits'), ccmatrix_combined_shifted, overwrite=True)

    axes, cbar = plot_ccmatrix(ccmatrix_combined, ccmatrix_combined_shifted, rv_sample, phase_combined)
    plt.suptitle(suptitle, size=12)
    plt.savefig(os.path.join(dirout, 'ccmatrix_bestfit_sigma.png'), dpi=200)
    plt.close()

    _ = ttest_on_trails(ccmatrix_combined_shifted, trail_width=TTEST_TRAIL_WIDTH, radius=TTEST_OUT_OF_TRAIL_RADIUS, plot=True)
    plt.suptitle(suptitle, size=12)
    plt.savefig(os.path.join(dirout, 'ttest_bestfit_sigma.png'), dpi=200)
    plt.close()



    # Expected (vsys, Kp) from literature.
    suptitle = r'Average C-C Matrix (Expected planet params: vsys={:.2f} km/s, Kp={:.2f} km/s)'.format(systemparams['vsys']/1e3, systemparams['kp']/1e3)
    rvplanet = -vbary_combined + systemparams['vsys'] + systemparams['kp']*np.sin(phase_combined*2*np.pi) # rv trial, shifted to the planet's rest frame
    ccmatrix_combined_shifted = shift_to_new_restframe(ccmatrix_combined, rv0=rv_sample, rvshift=rvplanet)
    fits.writeto(os.path.join(dirout, 'ccmatrix_shifted_expected_planet_params.fits'), ccmatrix_combined_shifted, overwrite=True)

    axes, cbar = plot_ccmatrix(ccmatrix_combined, ccmatrix_combined_shifted, rv_sample, phase_combined)
    plt.suptitle(suptitle, size=12)
    plt.savefig(os.path.join(dirout, 'ccmatrix_expected_planet_params.png'), dpi=200)
    plt.close()

    _ = ttest_on_trails(ccmatrix_combined_shifted, trail_width=TTEST_TRAIL_WIDTH, radius=TTEST_OUT_OF_TRAIL_RADIUS, plot=True)
    plt.suptitle(suptitle, size=12)
    plt.savefig(os.path.join(dirout, 'ttest_expected_planet_params.png'), dpi=200)
    plt.close()

loglike_f = os.path.join(dirout_significances+'/bl19_gridsearch/', 'loglike.fits')
 

print('Done.')

te = time.time()
ti = te-ts
print(f'Total time elapsed: {ti} s')