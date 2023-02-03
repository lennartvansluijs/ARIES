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

ARIES_BASE_DIR = '../..'
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


# targetname = 'wasp33'
# obsdates = ['20161015', '20161019', '20161020']
# dirname_models = "w33_gcm_elsie"
# model_fname = 'phase_dependent.txt'
# phase_filter_mode = 'out_of_occultation'
# dirname_residuals = "pca_7iter_masked_hpf"
# template_mode = '2D'
# continuum_subtracted = False

# targetname = 'wasp33'
# obsdates = ['20161015', '20161019', '20161020']
# dirname_models = "w33_gcm_elsie"
# model_fname = 'Em_0.0_template_CO.txt'
# phase_filter_mode = 'in_full_occultation'
# dirname_residuals = "pca_7iter_masked_hpf"
# template_mode = '1D'
# continuum_subtracted = False


# targetname = 'wasp33'
# obsdates = ['20161015', '20161019', '20161020']
# dirname_models = "w33_aries_josh_selfconsistent"
# model_fname = "WASP-33.redist=0.5.pp.z10.more.hires.7.txt"
# phase_filter_mode = 'out_of_occultation'
# dirname_residuals = "pca_7iter_masked_hpf"
# template_mode = '1D'
# continuum_subtracted = False

parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdates', type=str)
parser.add_argument('-dirname_models', type=str)
parser.add_argument('-fname_model', type=str)
parser.add_argument('-dirname_residuals', type=str)
parser.add_argument('-phase_filter_mode', type=str)
parser.add_argument('-template_mode', type=str)
parser.add_argument('-apply_hpf_to_model', type=bool)
parser.add_argument('-run_mode', type=str)
parser.add_argument('-use_crosscorr_weighting_scheme', type=bool, default=True)
parser.add_argument('-orders', type=str)


args = parser.parse_args()
targetname = args.targetname
obsdates = args.obsdates.split(' ')
dirname_models = args.dirname_models
model_fname = args.fname_model
dirname_residuals = args.dirname_residuals
template_mode = args.template_mode
phase_filter_mode = args.phase_filter_mode
apply_hpf_to_model = args.apply_hpf_to_model
run_mode = args.run_mode
use_crosscorr_weighting_scheme = args.use_crosscorr_weighting_scheme
orders = np.array(args.orders.split(' '), dtype=int)
systemparams = systems[targetname_to_key(targetname)]

continuum_subtracted = False
if apply_hpf_to_model:
    model_extension = '_hpf'
else:
    model_extension = ''

# targetname = 'wasp33'
# obsdates = ['20161015', '20161019', '20161020']
# dirname_models = "w33_expanded_grid_noH2O"
# model_fname = 'wasp33_aries_grid_t3_2000_p1_3_p3_6_a2_0d170000_z_1d00000_expanded_noH2O_7_spec.csv'
# phase_filter_mode = 'out_of_occultation'
# dirname_residuals = "pca_7iter_masked_hpf"
# template_mode = '1D'
# continuum_subtracted = False



# # include vrot of WASP-33 b
# targetname = 'wasp33'
# obsdates = ['20161015', '20161019', '20161020']
# dirname_models = "w33_aries_josh_selfconsistent_vrot"
# model_fname = "WASP-33.redist=0.5.pp.z10.more.hires.7.txt"
# phase_filter_mode = 'out_of_occultation'
# dirname_residuals = "pca_7iter_masked_hpf"
# template_mode = '1D'
# continuum_subtracted = False



dirin_models = os.path.abspath(ARIES_BASE_DIR + f"models/{targetname}/{dirname_models}")
#model_fname = "phase_dependent.txt"
fpath_model = os.path.join(dirin_models, model_fname)
NO_TXT_EXTENSION = slice(0,-4,1)

if len(obsdates) == 1:
    significances_basedir = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/{obsdates[0]}/significances/')
elif len(obsdates) == 3:
    significances_basedir = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/all/significances/')
else:
    raise ValueError('Either use one observing day or combine all three days.')

if continuum_subtracted:
    dirout_significances = os.path.join(
        significances_basedir,
        f'/{dirname_models}/{dirname_residuals}/{model_fname[NO_TXT_EXTENSION]}_continuum_subtracted/{phase_filter_mode}')
else:
    dirout_significances = os.path.join(
        significances_basedir,
        f'{dirname_models}/{dirname_residuals}/{model_fname[NO_TXT_EXTENSION]}{model_extension}/{phase_filter_mode}'
    )

#dirout_significances += '_x1_injected'
# if is_injected:
#     dirout_significances = os.path.abspath(ARIES_BASE_DIR + f"/data/{targetname}/all/processed/injected/{args.dirout_significances}")
if not os.path.exists(dirout_significances):
    os.makedirs(dirout_significances)


run_bl19_multinest = bool(int(run_mode[0]))
run_bl19_gridsearch = bool(int(run_mode[1]))
run_crosscorr = bool(int(run_mode[2]))
do_snr_for_all_nights = bool(int(run_mode[3]))
run_wilks_theorem = bool(int(run_mode[4]))

TTEST_TRAIL_WIDTH = 3
TTEST_OUT_OF_TRAIL_RADIUS = 5

nx, ny = (31, 31)
dvsys_all = np.linspace(-100e3, 100e3, nx)
dkp_all = np.linspace(-150e3, 150e3, ny)

RV_MIN = -1000e3 # m/s
RV_MAX = 1000e3 # m/s
DELTA_RV = 5e3 
nshifts = int((RV_MAX-RV_MIN)/DELTA_RV + 1)
rv_sample = np.linspace(RV_MIN, RV_MAX, nshifts)

result = {'snr':{}, 'ttest':{}, 'logL_grid':{}, 'logL_multinest':{}} # allocate memory to save results

# In[32]:



# Load all parameters up to point of running loglike.
spectralorders_all = []
templateorders_all = []

for obsdate in obsdates:
    #  Define observing date specific input directories
    OBSERVATION_BASE_DIR = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}')
    dirin_meta = os.path.abspath(OBSERVATION_BASE_DIR+'/meta')
    dirin_residuals = os.path.abspath(OBSERVATION_BASE_DIR+f'/processed/spectra/detrending/{dirname_residuals}')
    #     else:
    #         dirin_residuals = os.path.abspath(ARIES_BASE_DIR + f"/data/{targetname}/{obsdate}/processed/injected/{args.dirin_data}")
    
    #     dirin_residuals = f'/home/lennart/measure/data/wasp33/{obsdate}/processed/injected/w33_aries_josh_selfconsistent/WASP-33.redist=0.5.pp.z10.more.hires.7/x1/detrending/pca_7iter_masked_hpf'
    
    dirin_templates = os.path.abspath(OBSERVATION_BASE_DIR + f'/processed/hrccs_templates/{dirname_models}/' + os.path.basename(fpath_model)[NO_TXT_EXTENSION]) + model_extension

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


# In[6]:

if run_bl19_gridsearch or run_bl19_multinest:
    if template_mode == '1D':
        h = BrogiLineBayesianFramework_1DTemplate(spectralorders, templateorders, planetary_system=systemparams)
    elif template_mode == '2D':
        h = BrogiLineBayesianFramework_2DTemplate(spectralorders, templateorders, planetary_system=systemparams)
    else:
        raise ValueError('Unknown template mode: {}'.format(template_mode))


if run_bl19_multinest:
    h.run_multinest(dirout=dirout_significances+f'/bl19_multinest/', resume=False, verbose=True)
    h.plot_multinest(dirin=dirout_significances+f'/bl19_multinest/')
    plt.close()
    
if run_bl19_gridsearch:
    print('\n\tRunning loglike gridsearch...')
    
    
    # Try to load best scaling parameter "a"
    f = os.path.join(dirout_significances+f'/bl19_multinest/','stats.json')
    try:
        p = load_maxaposterior(f)
        loga_max = p[2]
    except:
        warnings.warn(f'Unable to load scaling parameter, setting "a" equal to 1. Scaling may be wrong.')
        loga_max = 0.
    
    h.run_gridsearch(dirout=dirout_significances+f'/bl19_gridsearch/',
    dvsys_all=dvsys_all, dkp_all=dkp_all, loga=loga_max)
    h.plot_gridsearch(dirin=dirout_significances+f'/bl19_gridsearch/')
    plt.close()



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
    obsdateid_list = []
    for ndate, obsdate in enumerate(np.unique([so.obsdate for so in spectralorders])):
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
        if use_crosscorr_weighting_scheme:
            dirin_weights = ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}/processed/weights/{dirname_models}'
            fname = f'{obsdate}_{model_fname[NO_TXT_EXTENSION]}{model_extension}_cc_weights.txt'
            data = np.loadtxt(os.path.join(dirin_weights, fname))
            cc_weights = np.array([w for norder, w in zip(data[0,:], data[1,:]) if norder in orders])
        else:
            cc_weights = np.ones(len(spectralorders_obsdate))
        ccmatrix_avg = np.average(ccmatrices, weights=cc_weights, axis=0)
        fits.writeto(os.path.join(dirout, '{}_ccmatrix_avg_{}.fits'.format(so.target, obsdate)), ccmatrix_avg, overwrite=True)

        # Create an overview plot of all cross-correlation results for all orders
        axes = plot_ccmatrices(ccmatrices, ccmatrix_avg, np.zeros(ccmatrix_avg.shape), orders, rv_sample)
        plt.savefig(os.path.join(dirout, f'{targetname}_overview_ccmatrices_{obsdate}.png'), dpi=200)
        plt.close()
        
        ccmatrices_allnights.append(ccmatrix_avg)
        nframes = len(spectralorders_obsdate[0].phase)
        obsdateid_list.append([ndate for i in range(nframes)])
        phase_allnights.append(spectralorders_obsdate[0].phase)
        vbary_allnights.append(spectralorders_obsdate[0].vbary)
        
        
    ccmatrix_allnights = np.concatenate(np.array(ccmatrices_allnights))
    phase_allnights = np.concatenate(np.array(phase_allnights))
    vbary_allnights = np.concatenate(np.array(vbary_allnights))
    obsdateid_list = np.concatenate(np.array(obsdateid_list))
    ind_sorted = phase_allnights.argsort()
    phase_combined = phase_allnights[ind_sorted]
    vbary_combined = vbary_allnights[ind_sorted]
    ccmatrix_combined = ccmatrix_allnights[ind_sorted, :]
    obsdateid_combined = obsdateid_list[ind_sorted]
    
    # save ccmatrix
    fits.writeto(os.path.join(dirout, f'{targetname}_ccmatrix_all.fits'), ccmatrix_combined, overwrite=True)
    np.save(os.path.join(dirout, f'{targetname}_ccmatrix_all_rv.npy'), rv_sample)
    np.save(os.path.join(dirout, f'{targetname}_ccmatrix_all_phase.npy'), phase_combined)
    np.save(os.path.join(dirout, f'{targetname}_ccmatrix_all_vbary.npy'), vbary_combined)
    
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
    
    # Create trail plots for all nights only
    axes = plot_detection_matrix(snrmatrix, dkp_all=dkp_all, dvsys_all=dvsys_all,
                                 kp=systemparams['kp'], vsys=systemparams['vsys'],
                                 title='Cross-correlation (SNR)', mode='snr')
    plt.savefig(os.path.join(dirout, f'snrmatrix.png'), dpi=200)
    plt.close()

    axes = plot_detection_matrix(sigmamatrix, dkp_all=dkp_all, dvsys_all=dvsys_all,
                                 kp=systemparams['kp'], vsys=systemparams['vsys'],
                                 title='Cross-correlation (T-test)', mode='ttest')
    plt.savefig(os.path.join(dirout, f'ttestmatrix.png'), dpi=200)
    plt.close()
        
    if do_snr_for_all_nights:
        # Create S/N plot for all individual nights
        for ndate, obsdate in enumerate(obsdates):
            # mask phases outside of range
            selected_phases_combined = phase_filter(
                phase_obs=phase_combined,
                occultation_phases=systemparams["occultation_phases"],
                mode=phase_filter_mode
            )
            # only selected phases of some nights
            is_correct_night = (obsdateid_combined == ndate)
            selected_phases_combined = np.logical_and(selected_phases_combined, is_correct_night)

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
            fname = os.path.join(dirout, f'snrmatrix_{obsdate}.fits')
            fits.writeto(fname, snrmatrix, overwrite=True)
            fname = os.path.join(dirout, f'sigmamatrix_{obsdate}.fits')
            fits.writeto(fname, sigmamatrix, overwrite=True)
            fname = os.path.join(dirout, f'grid_{obsdate}.txt')
            header = f"center: (vsys, kp) = ({systemparams['vsys']}, {systemparams['kp']}) \n"
            "grid: dvsys_all (m/s) | dkp_all (m/s)"
            np.savetxt(fname, np.c_[dvsys_all, dkp_all], header=header, delimiter=',')
            center = (systemparams['vsys'], systemparams['kp'])
            with open(os.path.join(dirout,'center.pickle'), 'wb') as f:
                pickle.dump(center, f)
    
            # Create trail plots for all nights only
            axes = plot_detection_matrix(snrmatrix, dkp_all=dkp_all, dvsys_all=dvsys_all,
                                         kp=systemparams['kp'], vsys=systemparams['vsys'],
                                        title='Cross-correlation (SNR)', mode='snr')
            plt.savefig(os.path.join(dirout, f'snrmatrix_{obsdate}.png'), dpi=200)
            plt.close()

            axes = plot_detection_matrix(sigmamatrix, dkp_all=dkp_all, dvsys_all=dvsys_all,
                                         kp=systemparams['kp'], vsys=systemparams['vsys'],
                                        title='Cross-correlation (T-test)', mode='ttest')
            plt.savefig(os.path.join(dirout, f'ttestmatrix_{obsdate}.png'), dpi=200)
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


    # Highest Loglike-matrix. Just slight change to the output directory here.
    f = os.path.join(dirout_significances+'/bl19_gridsearch/', 'loglike.fits')
    loglike = fits.getdata(f)

    vsys_max, kp_max, loglike_max = get_planet_params(loglike, trial_kp, trial_vsys)
    result['logL_grid'] = {'significance':loglike_max, 'vsys':vsys_max, 'kp':kp_max}
    suptitle=r'Average C-C Matrix (Best fit loglike: log L={:.0f}, $vsys$={:.2f} km/s, Kp={:.2f} km/s)'.format(loglike_max, vsys_max/1e3, kp_max/1e3)
    rvplanet = -vbary_combined + vsys_max + kp_max*np.sin(phase_combined*2*np.pi) # rv trial, shifted to the planet's rest frame
    ccmatrix_avg_shifted = shift_to_new_restframe(ccmatrix_combined, rv0=rv_sample, rvshift=rvplanet)
    fits.writeto(os.path.join(dirout, 'ccmatrix_shifted_bestfit_sigma.fits'), ccmatrix_avg_shifted, overwrite=True)

    axes, cbar = plot_ccmatrix(ccmatrix_combined, ccmatrix_combined_shifted, rv_sample, phase_combined)
    plt.suptitle(suptitle, size=12)
    plt.savefig(os.path.join(dirout_significances+'/bl19_gridsearch/', 'ccmatrix_bestfit_loglike.png'), dpi=200)
    plt.close()

    _ = ttest_on_trails(ccmatrix_combined_shifted, trail_width=TTEST_TRAIL_WIDTH, radius=TTEST_OUT_OF_TRAIL_RADIUS, plot=True)
    plt.suptitle(suptitle, size=12)
    plt.savefig(os.path.join(dirout_significances+'/bl19_gridsearch/', 'ttest_bestfit_loglike.png'), dpi=200)
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
if run_wilks_theorem and os.path.exists(loglike_f):        
    print("\n\tApplying Wilks' Theorem...")
    dirout_wilks = dirout_significances+'/wilks/'
    if not os.path.exists(dirout_wilks):
        os.mkdir(dirout_wilks)

    # Highest Loglike-matrix. Just slight change to the output directory here.
    f = os.path.join(dirout_significances+'/bl19_gridsearch/', 'loglike.fits')
    logL = fits.getdata(f)

    # Plot 2 x delta logL
    dlogL = 2*(logL-logL.max())
    axes = plot_detection_matrix(dlogL,
                                 dkp_all=dkp_all,
                                 dvsys_all=dvsys_all,
                                 kp=systemparams['kp'], vsys=systemparams['vsys'],
                                 title=f'WASP-33 b: 2$\Delta$log(L) (night={obsdate[0:4]}/{obsdate[4:6]}/{obsdate[6:]})',
                                 mode='loglike')
    fits.writeto(os.path.join(dirout_wilks, '2deltalogL.fits'), dlogL, overwrite=True)
    plt.savefig(os.path.join(dirout_wilks, '2deltalogL.pdf'))
    plt.savefig(os.path.join(dirout_wilks, '2deltalogL.png'), dpi=200)
    plt.close()


    # 2 * Delta log L follows chi2 with dof equal to nparams len([kp, vsys])=2
    degrees_of_freedom = 2
    wilksmap = np.zeros(logL.shape)
    for i in range(logL.shape[0]):
        for j in range(logL.shape[1]):
            p_value = stats.chi2.sf(-dlogL[i,j], degrees_of_freedom)
            wilksmap[i,j] = stats.norm.isf(p_value/2.) # convert p-value into corresponding sigma-value

    try:
        axes = plot_detection_matrix(wilksmap, dkp_all=dkp_all, dvsys_all=dvsys_all,kp=systemparams['kp'], vsys=systemparams['vsys'],
                                         title=f"Detection significance (Wilks' theorem) (night={obsdate[0:4]}/{obsdate[4:6]}/{obsdate[6:]})", mode='wilks2')

        fits.writeto(os.path.join(dirout_wilks, 'wilksmap.fits'), wilksmap, overwrite=True)
        plt.savefig(os.path.join(dirout_wilks, 'wilksmap.pdf'))
        plt.savefig(os.path.join(dirout_wilks, 'wilksmap.png'), dpi=200)
        plt.close()
    except:
        warnings.warn('Failed to plot wilksmap.')

    print('Done.')
else:
    warnings.warn("No loglike gridsearch result found. Skipping Wilk's theorem.")
    
print('\n\tWriting result to file...')

with open(os.path.join(dirout_significances, 'result.pickle'), 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done.')

te = time.time()
ti = te-ts
print(f'Total time elapsed: {ti} s')