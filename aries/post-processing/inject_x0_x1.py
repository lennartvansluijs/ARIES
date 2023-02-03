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

from aries.cleanspec import SpectralOrder, TemplateOrder, clip_mask
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
from aries.cleanspec import simulate_planet_spectrum

import subprocess
from tqdm import tqdm


# In[29]:

# start timer
ts = time.time()


# In[30]:


# Get target info from header arguments
parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdates', type=str)
parser.add_argument('-dirname_models', type=str)
parser.add_argument('-fname_model', type=str)
parser.add_argument('-phase_filter_mode', type=str)
parser.add_argument('-dirname_residuals', type=str)
parser.add_argument('-apply_hpf_to_model', type=bool)
parser.add_argument('-orders', type=str)
parser.add_argument('-ninj', type=int)
parser.add_argument('-wavcal_mode')
parser.add_argument('-alignment_mode')
parser.add_argument('-voffset_list')
parser.add_argument('-ninj_list')
parser.add_argument('-phase_inj')
parser.add_argument('-modelname_extension')

args = parser.parse_args()
targetname = args.targetname
obsdates = args.obsdates.split(' ')
dirname_models = args.dirname_models
model_fname = args.fname_model
phase_filter_mode = args.phase_filter_mode
dirname_residuals = args.dirname_residuals
apply_hpf_to_model = args.apply_hpf_to_model
systemparams = systems[targetname_to_key(targetname)]
orders = np.array(args.orders.split(' '), dtype=int)
ninj = args.ninj
wavcal_mode = args.wavcal_mode
voffset_list = args.voffset_list
alignment_mode = args.alignment_mode
modelname_extension = args.modelname_extension
phase_inj = args.phase_inj
if wavcal_mode == None:
    wavcal_mode = 'wavcal'
if alignment_mode == None:
    alignment_mode = 'alignment'
if voffset_list == None:
    voffset_list = np.array([(0., 0.)])
if ninj == None:
    ninj_list = np.array([0,ninj]) # injection strengths
else:
    ninj_list = np.array([0,1]) # injection strengths
if modelname_extension == None:
    modelname_extension = ''

# targetname = 'wasp33'
# obsdates = ['20161015', '20161019', '20161020']
# dirname_models = "w33_gcm_elsie"
# model_fname = 'Em_0.0_template_OH.txt'
# phase_filter_mode = 'out_of_occultation'
# dirname_residuals = "pca_7iter_masked_hpf"
# template_mode = '2D'
# continuum_subtracted = False


dirin_models = os.path.abspath(ARIES_BASE_DIR + f"/models/{targetname}/{dirname_models}")
#model_fname = "phase_dependent.txt"
fpath_model = os.path.join(dirin_models, model_fname)

NO_TXT_EXTENSION = slice(0,-4,1)


if len(obsdates) == 1:
    significances_basedir = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/{obsdates[0]}/significances/')
    print('Using single observing night.')
elif len(obsdates) > 1:
    significances_basedir = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/all/significances/')
    print('Using all observing nights.')
else:
    raise ValueError('Either use one observing day or combine all three days.')

if apply_hpf_to_model:
    model_extension = modelname_extension + '_hpf'
else:
    model_extension = modelname_extension + ''

dirout_significances = os.path.join(significances_basedir,
                                    f'{dirname_models}/{dirname_residuals}/{model_fname[NO_TXT_EXTENSION]}{model_extension}/{phase_filter_mode}')
if not os.path.exists(dirout_significances):
    os.makedirs(dirout_significances)


# injection parameters
do_injection = True
do_detrending = True
do_significances = True



# ----
# Load aligned data = templates
# ---


# Load all parameters up to point of running loglike.
spectralorders_all = []
templateorders_all = []

for obsdate in obsdates:
    #  Define observing date specific input directories
    OBSERVATION_BASE_DIR = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}')
    dirin_meta = os.path.abspath(OBSERVATION_BASE_DIR+'/meta')
    dirin_data = os.path.abspath(OBSERVATION_BASE_DIR+f'/processed/spectra/{alignment_mode}/')
    dirin_templates = os.path.abspath(OBSERVATION_BASE_DIR + f'/processed/hrccs_templates/{dirname_models}/' + os.path.basename(fpath_model)[NO_TXT_EXTENSION])

    dirin_wavcal = os.path.abspath(OBSERVATION_BASE_DIR+f'/processed/spectra/{wavcal_mode}')
    if apply_hpf_to_model:
        dirin_templates += '_hpf'

    print('\nInput parameters:')
    print(f'\tDate\t: {obsdate}')
    print(f'\tResiduals input directory\t: {dirin_data}')
    print(f'\tMeta data input directory\t: {dirin_meta}')
    print(f'\tSignificances output directory\t: {dirout_significances}')
    print(f'\tTemplates input directory\t: {dirin_templates}')
    print(f'\tInjection strength\t: x{ninj}')
    #  Load barycentric corrected time stamps and orbital phase
    print('\n\tLoading observed times, phases and barycentric data...')
    
    fpath_times = os.path.join(dirin_meta, 'science_times.txt')
    times_bjdtbd = np.loadtxt(fpath_times, dtype='str', delimiter=',').T[4].astype('float')
    fpath_vbary = os.path.join(dirin_meta, 'vbarycorr.txt')
    phase, vbary, rvplanet = np.loadtxt(fpath_vbary)
    print('Done.')
    
    
    #  Load residuals
    print('\n\tLoading observed residuals...')
    
    data_all = [fits.getdata(os.path.join(dirin_data, f'order_{norder}/{targetname}_order_aligned.fits')) for norder in orders]
    mask_all = [fits.getdata(os.path.join(dirin_data, f'order_{norder}/{targetname}_order_mask.fits')) for norder in orders]
    wavsolution_all = [np.loadtxt(os.path.join(dirin_wavcal, '{}_wavcal_order_{}_wavsolution.txt'.format(targetname, norder))).T for norder in orders]
    
    spectralorders = []
    for (norder, data, mask, wavsolution) in zip(orders, data_all, mask_all, wavsolution_all):
        data_c = clip_mask(data, mask)
        mask_c = np.zeros(data_c.shape)
        so = SpectralOrder(
            data=data_c,
            mask=mask_c,
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

# #  Apply mask to frames outside of selection
# for so in spectralorders:
#     selected_phases = phase_filter(
#         phase_obs=so.phase,
#         occultation_phases=systemparams["occultation_phases"],
#         mode=phase_filter_mode
#     )
#     so.mask[~selected_phases,:] = True

# #  Remove fully masked orders
# spectralorders_s = []
# templateorders_s = []
# for to, so in zip(templateorders, spectralorders):
#     if not np.all(so.mask):
#         spectralorders_s.append(so)
#         templateorders_s.append(to)
# spectralorders = spectralorders_s
# templateorders = templateorders_s

if phase_inj == None:
    phase_inj = spectralorders[0].phase

# ----
# Do injection
# ----
if do_injection:
    
    
    for obsdate in obsdates:
        
        
        dirout_injected = ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}/processed/injected/{dirname_models}'
        
        if not os.path.exists(dirout_injected):
            os.makedirs(dirout_injected)
            
            
        
        print('\n\tPreparing model injection into the data...')
        print(f'Obsdate={obsdate}')
        for dvsys_trial, dkp_trial in voffset_list:
            for ninj in ninj_list:
                    dirname = os.path.abspath(os.path.join(dirout_injected, f'{model_fname[NO_TXT_EXTENSION]}{model_extension}', f'x{ninj}'))
                    if dvsys_trial != 0:
                        dirname += (f'_dvsys={int(dvsys_trial/1e3)}')
                    if dkp_trial != 0:
                        dirname += (f'_dkp={int(dkp_trial/1e3)}')
                    

                    spectralorders_during_obsdate = [so for so in spectralorders if so.obsdate == obsdate]
                    templateorders_during_obsdate = [to for to in templateorders if to.obsdate == obsdate]
                    for niter, (so, to) in enumerate(zip(spectralorders_during_obsdate, templateorders_during_obsdate), 1):
                        rv_inj = get_rvplanet(-so.vbary, systemparams['vsys']+dvsys_trial, systemparams['kp']+dkp_trial, phase_inj)

                        template_wav = to.wavegrid
                        template_model = to.data.T
                        model = simulate_planet_spectrum(template_wav, template_model, data_wav=so.wavsolution[1], rvplanet=rv_inj, mode='2D') # 10**template spec as model in 10log(F), template wav already in micron
                        dirout_order = dirname + f'/aligned_injected/order_{so.norder}'
                        if not os.path.exists(dirout_order):
                            os.makedirs(dirout_order)

                        print(f'\tInjecting template into spectral time series ({niter}/{len(spectralorders_during_obsdate)})')
                        #print(so.norder, so.data.shape, model.shape)
                        data_inj = so.data * (1. + ninj*model)
                        

                        so.plot(data=so.data, yunit='phase', xunit='micron')
                        plt.savefig(os.path.join(dirout_order, '{}_data_noinj.png'.format(targetname, so.norder)), dpi=200)
                        fits.writeto(os.path.join(dirout_order, '{}_data_noinj.fits'.format(targetname, so.norder)), so.data, overwrite=True)
                        plt.close()

                        so.plot(data=ninj*model, mask=np.zeros(model.shape), xunit='micron', yunit='phase',
                                figtitle='x{} template model shifted to vplanet (order={})'.format(ninj, so.norder))
                        fits.writeto(os.path.join(dirout_order, '{}_model_order_{}.fits'.format(targetname, so.norder)), model, overwrite=True)
                        plt.savefig(os.path.join(dirout_order, '{}_model_order_{}.png'.format(targetname, so.norder)), dpi=200)
                        plt.close()

                        so.plot(data=data_inj, xunit='micron', yunit='phase',figtitle='Data with model (x{}) injected (order={})'.format(ninj, so.norder))
                        fits.writeto(os.path.join(dirout_order, '{}_data_inj.fits'.format(targetname, so.norder)), data_inj, overwrite=True)
                        fits.writeto(os.path.join(dirout_order, '{}_data_inj_mask.fits'.format(targetname)), so.mask, overwrite=True)
                        plt.savefig(os.path.join(dirout_order, '{}_data_inj.png'.format(targetname)), dpi=200)
                        plt.close()
                        
        print('Done.')
        
        
# ---
# Do detrending
# ---

if do_detrending:
    for obsdate in obsdates:
        dirin_injected = ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}/processed/injected/{dirname_models}'
        model_injected_list = np.array([f'x{ninj}' for ninj in ninj_list])
        with tqdm(total=len(model_injected_list), desc='detrending') as pbar:
            for model_injected in model_injected_list:
                dirin_data = os.path.join(dirin_injected, f'{model_fname[NO_TXT_EXTENSION]}{model_extension}', model_injected, 'aligned_injected')
                parentdirout = os.path.join(dirin_injected, f'{model_fname[NO_TXT_EXTENSION]}{model_extension}', model_injected, 'detrending')
                cmd = ['python', os.path.abspath(ARIES_BASE_DIR + '/aries/post-processing/do_cleanspec.py'), targetname, '-obsdate', obsdate, '-dirin_data', dirin_data, '-parentdirout', parentdirout, '-orders', " ".join(np.array(orders, dtype=str)), '-wavcal_mode', wavcal_mode]
                subprocess.call(cmd)
                pbar.update()


if do_significances:
    dirin_data_list = []
    for dvsys_trial, dkp_trial in voffset_list:
            for ninj in ninj_list:
                dirname = f'{dirname_models}/{model_fname[NO_TXT_EXTENSION]}{model_extension}/x{ninj}/detrending/{dirname_residuals}'
                if dvsys_trial != 0: dirname += (f'_dvsys={int(dvsys_trial/1e3)}')
                if dkp_trial != 0: dirname += (f'_dkp={int(dkp_trial/1e3)}')
                dirin_data_list.append(dirname)
    t0 = time.time()
    
    dirin_output_list = []
    for dvsys_trial, dkp_trial in voffset_list:
            for ninj in ninj_list:
                dirname = f'{dirname_models}/{model_fname[NO_TXT_EXTENSION]}{model_extension}/x{ninj}/significances/{dirname_residuals}'
                if dvsys_trial != 0: dirname += (f'_dvsys={int(dvsys_trial/1e3)}')
                if dkp_trial != 0: dirname += (f'_dkp={int(dkp_trial/1e3)}')
                dirin_output_list.append(dirname)
    t0 = time.time()
    ntotal = len(dirin_output_list)
    for n, (dirin_data, dirin_output) in enumerate(zip(dirin_data_list, dirin_output_list), 1):
        print(f'\nIteration: {n}/{ntotal}')
        #cmd = ['python', os.path.abspath(ARIES_BASE_DIR + '/aries/post-processing/run_snr_and_likelihood.py'),'-targetname', targetname, '-obsdates', " ".join(obsdates), '-dirname_models', dirname_models, '-fname_model', model_fname, '-dirname_residuals', dirname_residuals, '-template_mode', '1D', '-phase_filter_mode', phase_filter_mode, "-apply_hpf_to_model", 'True', '-run_mode', '00100', '-use_crosscorr_weighting', 'False']
        cmd = [
            "python", os.path.abspath(ARIES_BASE_DIR+'/aries/post-processing/run_significances_injection.py'), "-model_fname", model_fname, "-dirin_data", dirin_data, "-dirout_significances", dirin_output, "-dirname_models", dirname_models, "-apply_hpf_to_model", str(apply_hpf_to_model), "-obsdates",  " ".join(obsdates), "-targetname", targetname, "-phase_filter_mode", phase_filter_mode, "-orders", " ".join(np.array(orders, dtype=str))
        ]
        subprocess.call(cmd)