# type following command in commandline:
# python kelt7_cleanspec.py 'kelt7' -obsdate '20161018' -dirin_data '/home/lennart/measure/data/kelt7/20161018/processed/spectra/alignment' -parentdirout '/home/lennart/measure/data/kelt7/20161018/processed/spectra/detrending'

#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import os
import sys
ARIES_BASE_DIR = '../..'
sys.path.append(ARIES_BASE_DIR)

sys.path.append(os.path.abspath(ARIES_BASE_DIR)+'/lib')

from astropy.time import Time
from astropy.io import fits
from astropy.constants import G, au, M_sun, R_sun, R_jup

from barycorrpy import utc_tdb, get_BC_vel, get_stellar_data
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from aries.constants import TABLEAU20
from aries.cleanspec import SpectralCube, SpectralOrder, clip_mask, pca_detrending
from aries.crosscorrelate import get_rvplanet
from aries.preprocessing import get_keyword_from_headers_in_dir, get_fits_fnames
from aries.crosscorrelate import calc_orbital_phase
from aries.crosscorrelate import estimate_kp
from aries.constants import ARIES_NORDERS, ARIES_NX
from aries.cleanspec import apply_highpass_filter, sliding_window_iter
from matplotlib import gridspec
import matplotlib.gridspec as gridspec


# ---
# Terminal header
# ---


sf, nc = 20, 30
print('\n'+'-'*(2*sf+nc))
print('-'*sf + '      MEASURE detrending     ' + '-'*sf)
print('-'*sf + '     by Lennart van Sluijs   ' + '-'*sf)
print('-'*(2*sf+nc))


# ---
# Detrending configuration parameters
# ---


OVERWRITE_DIROUT = True
DETRENDING_ALGORITHM = 'pca' # pick from 'bl19' or 'pca'
APPLY_DOWNWEIGHT_NOISY_COLS = False
NPOINTS_NORMALISATION = 50
CLIP_EDGES = True
LEFTMOST_PIXEL, RIGHTMOST_PIXEL = 50, 976
APPLY_MASK = True
MASK_SIGMA = 3.
MASK_WINDOW = 5
MASK_NBADS = 2
APPLY_HIGHPASSFILTER = True
HIGHPASSFILTER_FREQ_CUTOFF = 1./50. # pixel^-1
PCA_KMAX = 7
IDENTIFY_AND_CLIP_BADFRAMES = False
NBADFRAMES = 5
CLIP_FLAGGED_FNAMES = False


# ---
# Parse input parameters
# ---

parser = argparse.ArgumentParser()
parser.add_argument('targetname')
parser.add_argument('-obsdate')
parser.add_argument('-dirin_data')
parser.add_argument('-parentdirout')
parser.add_argument('-orders')
parser.add_argument('-wavcal_mode')

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate
dirin_data = args.dirin_data
parentdirout = args.parentdirout
orders = np.array(args.orders.split(' '), dtype=int)
wavcal_mode = args.wavcal_mode
if wavcal_mode == None:
    wavcal_mode = 'wavcal'

print('\nInput parameters:')
print(f'\tTargetname: {targetname}')
print(f'\tObsdate: {obsdate}')
print(f'\tData input directory name: {dirin_data}')
print(f'\tDetrended data ouput directory name: {parentdirout}')
print(f'\tWavcal mode: {wavcal_mode}')
if not os.path.exists(parentdirout):
    os.mkdir(parentdirout)

from aries.systemparams import systems, targetname_to_key
systemparams = systems[targetname_to_key(targetname)]
OBSERVATION_BASE_DIR = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}')

obsname = 'Multiple Mirror Telescope'

print('Done.')


# ---
# Create unique input directroy based on config parameters
# ---


methods = ''
if DETRENDING_ALGORITHM == 'pca':
    methods += f'{PCA_KMAX}iter_'
if APPLY_MASK:
    methods += 'masked_'
if APPLY_HIGHPASSFILTER:
    methods += 'hpf_'
if APPLY_DOWNWEIGHT_NOISY_COLS:
    methods += 'dnc_'
#methods += f'orders{orders[0]}-{orders[-1]}'
if methods.endswith('_'):
    methods = methods[:-1]

dirout_detrending = f'{DETRENDING_ALGORITHM}_{methods}'

# create new directory in case OVERWRITE_DIROUT = False
fullpath = lambda dirname: os.path.abspath(os.path.join(f'{parentdirout}/{dirname}/'))
dirname = np.copy(dirout_detrending)
i = 1
if not OVERWRITE_DIROUT:
    while os.path.isdir(fullpath(dirout_detrending)):
        dirout_detrending = f'{dirname}({i})'
        i += 1

dirout_detrending = fullpath(dirout_detrending)
if not os.path.exists(dirout_detrending):
    os.mkdir(dirout_detrending)

for norder in orders:
    dirout_order = os.path.join(dirout_detrending, f'order_{norder}')
    if not os.path.exists(dirout_order):
        os.mkdir(dirout_order)

dirin_wavcal = os.path.abspath(OBSERVATION_BASE_DIR+f'/processed/spectra/{wavcal_mode}')
dirin_meta = os.path.abspath(OBSERVATION_BASE_DIR+'/meta')

print(f'\n\tOutput location: {dirout_detrending}')
print(f'\tInput directory wavelength calibration: {dirin_wavcal}')
print(f'\tInput directory radial velocity + time meta data: {dirin_meta}')

print('Done.')


# ---
# Load barycentric corrected time stamps and orbital phase
# ---


print('\n\tLoading barycentric data...')


# fpath = os.path.join(dirin_meta, f'{targetname}_targetinfo.csv')
# with open(fpath, mode='r') as infile:
#     reader = csv.reader(infile)
#     systemparams = {rows[0]:float(rows[1]) for rows in reader}

fpath = os.path.join(dirin_meta, 'science_times.txt')
times_bjdtbd = np.loadtxt(fpath, dtype='str', delimiter=',').T[4].astype('float')

fpath = os.path.join(dirin_meta, 'vbarycorr.txt')
phase, vbary, rvplanet = np.loadtxt(fpath)
nobs = len(phase)
rvplanet_os = get_rvplanet(vbary, systemparams['vsys'], systemparams['kp'], phase=np.linspace(0,1,nobs)) # oversampled light curve

plt.figure(figsize=(5*1, 5*1.68))
plt.plot(rvplanet_os/1e3, np.linspace(0,1,nobs), color='k')
plt.scatter(rvplanet/1e3, phase, s=100, facecolors='none', edgecolors=TABLEAU20[0])
plt.ylim(0,1)
plt.ylabel('Orbital phase', size=12)
plt.xlabel('Velocity (km/s)', size=12)
plt.title(f"Phase coverage + planet RV (before barycorr) ({targetname.upper()})", size=12)
plt.close()

print('Done.')


# ---
# Load spectral orders data
# ---


print('\n\tLoading input spectral orders data...')

if os.path.basename(dirin_data) in ('aligned', 'alignment', 'alignment_with_stretching'):
    data_all = [fits.getdata(os.path.join(dirin_data, f'order_{norder}/{targetname}_order_aligned.fits')) for norder in orders]
    mask_all = [fits.getdata(os.path.join(dirin_data, f'order_{norder}/{targetname}_order_mask.fits')) for norder in orders]
    wavsolution_all = [np.loadtxt(os.path.join(dirin_wavcal, '{}_wavcal_order_{}_wavsolution.txt'.format(targetname, norder))).T for norder in orders]

elif os.path.basename(dirin_data) == 'aligned_injected':
    data_all = [fits.getdata(os.path.join(dirin_data, f'order_{norder}/{targetname}_data_inj.fits')) for norder in orders]
    mask_all = [fits.getdata(os.path.join(dirin_data, f'order_{norder}/{targetname}_data_inj_mask.fits')) for norder in orders]
    wavsolution_all = [np.loadtxt(os.path.join(dirin_wavcal, '{}_wavcal_order_{}_wavsolution.txt'.format(targetname, norder))).T for norder in orders]
else:
    raise NameError('Input directory name not understood.')

spectralorders = []
for (norder, data, mask, wavsolution) in zip(orders, data_all, mask_all, wavsolution_all):
    so = SpectralOrder(data=data, mask=mask, norder=norder, wavsolution=wavsolution, target=targetname, phase=phase)
    spectralorders.append(so)
    if not (so.data.shape[1] == len(so.wavsolution[0])):
        specc = np.arange(1, ARIES_NX+1)
        wmin, wmax = so.wavsolution[0][0], so.wavsolution[0][-1]
        wavmask = np.logical_and(specc >= wmin, specc <= wmax)
        so.data = so.data[:,wavmask]
        so.error = so.error[:,wavmask]
        so.mask = so.mask[:,wavmask]
        so.nobs, so.nx = so.data.shape
print('Done.')

# ---
# Normalisation
# ---


print('\n\tNormalise all spectra...')


def normalise_spectra_bl19(data, npoints=300):
    """Throughput correction as in Brogi & Line 2019."""
    nobs, npixels = data.shape
    data_n = np.zeros(shape=(nobs, npixels))
    for n in range(nobs):
        spec = data[n, :]
        brightest_n_points = np.sort(spec)[-npoints:]
        data_n[n, :] = spec/np.median(brightest_n_points)
    return data_n

for so in spectralorders:
    dirout_order = os.path.join(dirout_detrending, f'order_{so.norder}')
    data_n = normalise_spectra_bl19(so.data, npoints=NPOINTS_NORMALISATION)
    
    fpath = os.path.join(dirout_order, '{}_data_after_normalisation_{}.fits'.format(targetname, so.norder))
    fits.writeto(fpath, data_n, overwrite=True)
    so.plot(data_n, xunit='micron', yunit='phase', figtitle=f"After normalisation (BL'19) (order={so.norder})")
    fpath = os.path.join(dirout_order, '{}_data_after_normalisation_{}.png'.format(targetname, so.norder))
    plt.savefig(fpath, dpi=200)
    plt.close()
    
    plt.plot(np.mean(data_n, axis=0))
    plt.close()
    
    so.data = data_n

print('Done.')


# ---
# Detrending of quasi-stationary trends
# ---


# PCA detrending
print('\n\tDetrending data...')

if DETRENDING_ALGORITHM == 'pca':
    for n, so in enumerate(spectralorders, 1):
        dirout_order = os.path.join(dirout_detrending, f'order_{so.norder}')
        print('\tPCA detrending order: {} ({}/{})'.format(so.norder, n, len(orders)))
        data_detrended = pca_detrending(so.data, k=PCA_KMAX)

        fpath = os.path.join(dirout_order, '{}_data_before_pca_{}.fits'.format(targetname, so.norder))
        fits.writeto(fpath, so.data, overwrite=True)
        so.plot(data=so.data, figtitle=f"Before PCA (order={so.norder})")
        fpath = os.path.join(dirout_order, '{}_data_before_pca_{}.png'.format(targetname, so.norder))
        plt.savefig(fpath, dpi=300)
        plt.close()

        fpath = os.path.join(dirout_order, '{}_data_after_pca_{}.fits'.format(targetname, so.norder))
        fits.writeto(fpath, data_detrended, overwrite=True)
        so.plot(data=data_detrended, figtitle=f"After PCA (k={PCA_KMAX}) (order={so.norder})")
        fpath = os.path.join(dirout_order, '{}_data_after_pca_{}.png'.format(targetname, so.norder))
        plt.savefig(fpath, dpi=300)
        plt.close()
        
        so.data = data_detrended


print('Done.')

# ---
# Apply high pass filter
# ---

if APPLY_HIGHPASSFILTER:
    print('\n\tApplying high pass filter...')
    for so in spectralorders:
        dirout_order = os.path.join(dirout_detrending, f'order_{so.norder}')
        
        fpath = os.path.join(dirout_order, '{}_data_before_hpf_{}.fits'.format(targetname, so.norder))
        fits.writeto(fpath, so.data, overwrite=True)
        so.plot(so.data, apply_mask=True, figtitle=f'Before highpass filter (order={so.norder})')
        fpath = os.path.join(dirout_order, '{}_data_before_hpf_{}.png'.format(targetname, so.norder))
        plt.savefig(fpath, dpi=300)
        plt.close()
        
        data_hpf = apply_highpass_filter(so.data, freq_cutoff=HIGHPASSFILTER_FREQ_CUTOFF)
        
        fpath = os.path.join(dirout_order, '{}_data_after_hpf_{}.fits'.format(targetname, so.norder))
        fits.writeto(fpath, data_hpf, overwrite=True)
        so.plot(data_hpf, apply_mask=True, figtitle=f'After highpass filter (order={so.norder})')
        fpath = os.path.join(dirout_order, '{}_data_after_hpf_{}.png'.format(targetname, so.norder))
        plt.savefig(fpath, dpi=300)
        plt.close()
        
        so.data = data_hpf.copy()

        
print('Done.')

# ---
# Clipping & masking
# ---

print('\n\tClipping and masking data...')

if CLIP_EDGES:
    for so in spectralorders:
        so.clip_edges(xmin=LEFTMOST_PIXEL, xmax=RIGHTMOST_PIXEL)
        so.plot()
        plt.close()

if APPLY_MASK:
    for so in spectralorders:
        data = so.data.copy()
        so.plot(apply_mask=False)
        plt.close()
        
        badpixelmap = np.array( abs(data - data.mean())/data.std() > MASK_SIGMA, dtype='int')
        
        # combine closely packed columns using a sliding window
        mask1d = np.array(badpixelmap.sum(axis=0) >= MASK_NBADS, dtype=int)
        new_mask = np.zeros(so.mask.shape)
        for i, w in enumerate(
        sliding_window_iter(mask1d, size=MASK_WINDOW)):
            if sum(w) >= np.floor(MASK_WINDOW/2.):
                new_mask[:,i:i+MASK_WINDOW] = 1.
                
        so.mask = new_mask
        so.plot(so.data,apply_mask=True)
        plt.close()
        
print('Done.')


# ---
# Save output
# ---

print('\n\tSaving output...')

params = [
    "dirin_data", "dirin_wavcal", "dirin_meta", "dirout_detrending",
    "OVERWRITE_DIROUT", "DETRENDING_ALGORITHM", "APPLY_DOWNWEIGHT_NOISY_COLS",
    "APPLY_MASK", "MASK_WINDOW", "MASK_SIGMA", "MASK_NBADS", "NPOINTS_NORMALISATION",
    "APPLY_HIGHPASSFILTER", "HIGHPASSFILTER_FREQ_CUTOFF", "PCA_KMAX", "CLIP_EDGES",
    "LEFTMOST_PIXEL", "RIGHTMOST_PIXEL"
]

settings = { key : eval(key) for key in params }

with open(os.path.join(dirout_detrending, 'settings.txt'), 'w') as f:
    for (k,v) in settings.items():
        f.write(f'{k}:{v}\n')

for so in spectralorders:    
    dirout_order = os.path.join(dirout_detrending, f'order_{so.norder}')
    # plot residual including mask
    so.plot(so.data, xunit='micron', figtitle=f'Residual (order={so.norder})')
    plt.savefig(os.path.join(dirout_order, f'{targetname}_masked_residual_{so.norder}.png'))
    plt.close()
    
    # save mask
    fpath = os.path.join(dirout_order, '{}_mask_{}.fits'.format(targetname, so.norder))
    fits.writeto(fpath, so.mask, overwrite=True)
    
    # save residual
    fpath = os.path.join(dirout_order, '{}_residual_{}.fits'.format(targetname, so.norder))
    fits.writeto(fpath, so.data, overwrite=True)
    
    # save current wavsolution
    fpathout = os.path.join(dirout_order, '{}_wavsolution_{}.txt'.format(targetname, so.norder))
    np.savetxt(fpathout, so.wavsolution)
    
print('Done.')