#!/usr/bin/env python
# coding: utf-8

# # Pre-processing: spectral extraction (WASP-33)
# ---

# <b> Modules and packages

# In[1]:


import argparse
import os
import pickle
import scipy
import shutil
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib import gridspec
import numpy as np

from astropy.io import fits
from settings import ARIES_BASE_DIR, DATA_BASE_DIR, REFERENCE_TARGET_SPEX, \
OVERWRITE_SPEX, CONFIG_BASE_DIR
import sys
CERES_BASE_DIR = os.path.abspath(ARIES_BASE_DIR+'/lib/ceres')
sys.path.append(ARIES_BASE_DIR)
sys.path.append(CERES_BASE_DIR+'/utils/Correlation')
sys.path.append(CERES_BASE_DIR+'/utils/GLOBALutils')
sys.path.append(CERES_BASE_DIR+'/utils/OptExtract')

import GLOBALutils
import Marsh

from aries.spectra import obtain_P, optimal_extraction
from aries.spectra import plot_spectral_orders_as_image
from aries.spectra import plot_spectral_orders
from aries.spectra import optext_all
from aries.spectra import make_spectral_cube_aries

from aries.preprocessing import plot_image
from aries.preprocessing import get_fits_fnames, load_imgs
from aries.traces import load_traces


# <b>Target info

# In[39]:


# Print to terminal
print('-'*50)
print('4. Extract spectra from Echelle traces')
print('-'*50)


# Get target info from header arguments
parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdate', type=str)

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate

OBSERVATION_BASE_DIR = os.path.abspath(DATA_BASE_DIR + '/{}/{}'.format(targetname, obsdate))


# <b>Algorithm parameters

# In[40]:


NPOOLS = 4
APERTURE_RADIUS = 6 # pixel
SIGMA_CLIP = 5
SIGMA_CLIP_COSMIC_RAYS = 5
PIXEL_INTERP_FRACTION = 0.4 # fraction of a pixel used for interpolation
MARSH_POLYDEG = 3 # degree of polynomials fitted by Marsh algorithm
VALUE_TO_MASK = np.nan


# ---

# ## Extract spectral orders for all science frames

# In[42]:




dirin_sciences = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr')
dirin_traces = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces')

sciences_fnames = np.sort(get_fits_fnames(dirin_sciences, key='science')) # sort to ensure times are in chronological order
sciences = load_imgs(dirin_sciences, sciences_fnames)

sciences_traces_fnames = ['trace_'+fname for fname in sciences_fnames]
sciences_traces = load_traces(dirin_traces, sciences_traces_fnames)

ntraces = sciences_traces[0].coefs_all.shape[0]

# Load spectral order ranges from configuration file
dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/meta')
if not os.path.exists(dirin):
    os.mkdir(dirin)

fpath = OBSERVATION_BASE_DIR+'/meta/{}_spectralextractionxranges.txt'.format(targetname)
if not os.path.exists(fpath) or OVERWRITE_SPEX:
    print('\tNo configuration file found, using default configuration file.')
    config_fpath = os.path.abspath(CONFIG_BASE_DIR + '/{}_spectralextraction_xranges.txt'.format(REFERENCE_TARGET_SPEX)) # Default config file
    shutil.copyfile(config_fpath, fpath)
else:
    print('\tConfiguration file found!')

config_fpath = OBSERVATION_BASE_DIR+'/meta/{}_spectralextractionxranges.txt'.format(targetname)
orders, xmin_all, xmax_all = np.loadtxt(config_fpath, dtype=int).T


if ntraces is not xmin_all.size:
    raise ValueError('Ntraces must be equal to number of traces in configuration file.')


OPTEXT_PARAMS = {
    'aperture' : APERTURE_RADIUS, # pixel
    'nsigma' : SIGMA_CLIP,
    'interp_fraction' : PIXEL_INTERP_FRACTION,
    'polydegree': MARSH_POLYDEG,
    'use' : 0, # 0 = Marsh, 1 = Horne
    'ncosmic' : SIGMA_CLIP_COSMIC_RAYS,
    'min_col' : xmin_all,
    'max_col' : xmax_all
}


# In[ ]:


dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/optext')
if not os.path.exists(dirout):
    os.makedirs(dirout)

optext_all(sciences_fnames, sciences, sciences_traces,
           dirout=dirout, npools=NPOOLS, optext_params=OPTEXT_PARAMS, replace_bad_weights=True)


# ## Combine all extracted spectra into one datacube

# In[27]:


dirin_sciences = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr')
dirin_traces = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces')

sciences_fnames = np.sort(get_fits_fnames(dirin_sciences, key='science')) # sort to ensure times are in chronological order
sciences = load_imgs(dirin_sciences, sciences_fnames)
sciences_traces_fnames = ['trace_'+fname for fname in sciences_fnames]
sciences_traces = load_traces(dirin_traces, sciences_traces_fnames)


# In[37]:


dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/optext')
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra')

fpaths_result_optext = [os.path.join(dirin, 'result_optext_'+f+'.fits') for f in sciences_fnames]
sc = make_spectral_cube_aries(fpaths_result_optext)
sc.apply_mask(apply_to_value=VALUE_TO_MASK)
sc.target = targetname.upper()
sc.plot(vmin=0, vmax=np.quantile(sc.data.flatten(), 0.997))
fpath = os.path.join(dirout, 'spectral_time_series_all_orders_{}'.format(targetname))
plt.savefig(fpath+'.png', dpi=250)
sc.save(os.path.join(dirout, 'spectralcube_{}_raw.fits'.format(targetname)))
plt.close()


# ## Save as individual spectral orders

# In[33]:


dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/extracted')
if not os.path.exists(dirout):
    os.mkdir(dirout)

for n in range(1, sc.norders+1):
    so = sc.get_spectralorder(norder=n)

    if np.all(so.data == 0.): # order not properly extracted
        pass
    else:
        dirout_order = os.path.join(dirout, 'order_{}'.format(n))
        if not os.path.exists(dirout_order):
            os.mkdir(dirout_order)

    print('Saving spectral order {}/{}'.format(n, sc.norders))

    so.plot(so.data, vmin=0, vmax=np.quantile(so.data, 0.997),
            figtitle='Extracted Spectral Time Series, order={}, target={}'.format(so.norder, so.target.upper()))
    fname = dirout_order+'/{}_order_{}_extracted'.format(targetname, n)
    plt.savefig(fname+'.png', dpi=150)
    fits.writeto(fname+'.fits', so.data, overwrite=True)
    fname = dirout_order+'/{}_order_{}_mask'.format(targetname, n)
    fits.writeto(fname+'.fits', so.mask, overwrite=True)
    plt.close()
