#!/usr/bin/env python
# coding: utf-8

# # Pre-processing: flat defringing and correction

import argparse
import os
import pickle
import sys
from settings import ARIES_BASE_DIR, DATA_BASE_DIR
sys.path.append(ARIES_BASE_DIR)
PYTHON_VERSION = sys.version_info[0]
import scipy
import warnings

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import matplotlib
matplotlib.use('Agg')
matplotlib.cm.Greys_r.set_bad(color='black')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from aries.constants import get_tableau20_colors, TABLEAU20, ARIES_NX, ARIES_NY, ARIES_BADCOLUMNS

from aries.preprocessing import get_imgs_in_dir
from aries.preprocessing import get_master, get_elevation_from_fnames
from aries.preprocessing import plot_image
from aries.preprocessing import correct_badcolumns
from aries.preprocessing import make_badpixelmap
from aries.preprocessing import get_keyword_from_headers_in_dir
from aries.preprocessing import robust_polyfit
from aries.preprocessing import is_flat

from aries.flatcorr import fit_illumination_and_fringes
from aries.flatcorr import get_flats_fnames, load_traces, load_flats
from aries.flatcorr import dewarp_all_flats
from aries.flatcorr import fit_illumination_and_fringes_all_flats
from aries.flatcorr import warp_illumination_and_fringes_all_flats
from aries.flatcorr import correct_fringes
from aries.flatcorr import make_simple_badpixelmap, replace_badpixels

from aries.traces import plot_img_with_traces
from aries.traces import EchelleTraces, EchelleImageTransformer
from aries.traces import get_flats_and_traces, get_master_traces, plot_traces


# Print to terminal
print('-'*50)
print('3.b Flat fringes correction')
print('-'*50)


# Get target info from header arguments
parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdate', type=str)

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate

OBSERVATION_BASE_DIR = os.path.abspath(DATA_BASE_DIR + '/{}/{}'.format(targetname, obsdate))


# Algorithm parameters
BLAZE_FUNCTION_POLYDEGREE = 7
TRACE_APERTURE = 20
SIGMA_CLIP = 3.
FRINGE_FREQ_CUTOFF = 0.025
FLAT_KEYWORD = 'flat'
MASTER_FLAT_KEYWORD = 'master flat'
MASTER_DARK_KEYWORD = 'master_dark'
MASTER_METHOD = 'median'
OUTSIDE_TRACE_FILL_VALUE = np.nan
BADPIXEL_VMIN = 0.5
BADPIXEL_VMAX = 1.5


# Dewarping flat frames
parentdirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/')
if not os.path.exists(parentdirout):
    os.mkdir(parentdirout)

dirin_flats = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr')
dirin_traces = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces')
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/dewarped')
if not os.path.exists(dirout):
    os.mkdir(dirout)

flats_fnames = get_flats_fnames(dirin_flats)
traces_fnames = ['trace_' + fname for fname in flats_fnames]
flats = load_flats(dirin_flats, flats_fnames)
flats_traces = load_traces(dirin_traces, traces_fnames)

dewarp_all_flats(flats_fnames, flats, flats_traces, dirout)


# Illumination model and fringe model fitting
#
# Now let's create an illumination model and fringe model for all dewarped flats.
dirin_flats = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr')
dirin_flats_dewarped = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/dewarped')
dirin_traces = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces')
dirout_illumination = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/dewarped/illumination_models')
dirout_fringes = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/dewarped/fringes_models')

flats_fnames = get_flats_fnames(dirin_flats)
traces_fnames = ['trace_' + fname for fname in flats_fnames]
flats_dewarped_fnames = ['dewarped_' + fname for fname in flats_fnames]
flats_dewarped = load_flats(dirin_flats_dewarped, flats_dewarped_fnames)
flats_traces = load_traces(dirin_traces, traces_fnames)

fit_illumination_and_fringes_all_flats(flats_fnames, flats_dewarped, flats_traces, dirout_illumination, dirout_fringes,
                                      polydegree=BLAZE_FUNCTION_POLYDEGREE, aperture=TRACE_APERTURE,
                                      sigma=SIGMA_CLIP, freq_cutoff=FRINGE_FREQ_CUTOFF)


# Warp illumination and fringe model
dirin_flats = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr')
dirin_traces = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces')
dirin_illumination_models_dewarped = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/dewarped/illumination_models')
dirin_fringes_models_dewarped = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/dewarped/fringes_models')

dirout_illumination = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/illumination_models')
dirout_fringes = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/fringes_models')

flats_fnames = get_flats_fnames(dirin_flats)
traces_fnames = ['trace_' + fname for fname in flats_fnames]
illumination_models_dewarped_fnames = ['dewarped_illumination_model_' + fname for fname in flats_fnames]
fringes_models_dewarped_fnames = ['dewarped_fringes_model_' + fname for fname in flats_fnames]
illumination_models_dewarped = load_flats(dirin_illumination_models_dewarped, illumination_models_dewarped_fnames)
fringes_models_dewarped = load_flats(dirin_fringes_models_dewarped, fringes_models_dewarped_fnames)
flats_traces = load_traces(dirin_traces, traces_fnames)

warp_illumination_and_fringes_all_flats(flats_fnames, flats_traces, illumination_models_dewarped, fringes_models_dewarped, dirout_illumination, dirout_fringes)


# Flat fringe correction
dirin_flats = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr')
dirin_illumination = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/illumination_models')
dirin_fringes = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/fringes_models')

dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/fringescorr')

flats_fnames = get_flats_fnames(dirin_flats)
illumination_models_fnames = ['illumination_model_' + fname for fname in flats_fnames]
fringes_models_fnames = ['fringes_model_' + fname for fname in flats_fnames]
illumination_models = np.array(load_flats(dirin_illumination, illumination_models_fnames))
fringes_models = np.array(load_flats(dirin_fringes, fringes_models_fnames))
flats = np.array(load_flats(dirin_flats, flats_fnames))

correct_fringes(flats_fnames, flats, illumination_models, fringes_models, dirout)


# Flat badpixel map and correction
exptime = get_keyword_from_headers_in_dir('EXPTIME', dirin_flats, key=FLAT_KEYWORD)[0]
dirin_badpixelmap = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/badpixel_maps/')

fname = 'badpixelmap_master_dark_{}s.fits'.format(int(exptime))
fpath = os.path.join(dirin_badpixelmap, fname)
if os.path.exists(fpath):
    badpixelmap_darks = fits.getdata(fpath)
else:
    try:
        fname = [f for f in os.listdir(dirin_badpixelmap) if f.endswith('.fits') and MASTER_DARK_KEYWORD in f][0]
        fpath = os.path.join(dirin_badpixelmap, fname)
        badpixelmap_darks = fits.getdata(fpath)
    except:
        raise FileNotFoundError('No available master dark badpixel map found.')

dirin_flats_fringescorr = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/fringescorr')
flats_fringescorr = np.array(load_flats(dirin_flats_fringescorr, flats_fnames))

master_flat = get_master(flats_fringescorr, method=MASTER_METHOD)
master_flat_traces = get_master_traces(flats_traces, method=MASTER_METHOD)
master_illumination_model = get_master(illumination_models, method=MASTER_METHOD)

master_flat_corr = correct_badcolumns(master_flat, badcolumns=ARIES_BADCOLUMNS)
master_flat_normalized = master_flat_corr/master_illumination_model
outside_traces = np.isnan(master_illumination_model)
master_flat_normalized[outside_traces] = 1.

master_illumination_model[outside_traces] = 1.
badpixelmap = make_simple_badpixelmap(master_flat_normalized, BADPIXEL_VMIN, BADPIXEL_VMAX)
badpixelmap_combined = np.logical_or(badpixelmap_darks, badpixelmap)
master_flat_corr = replace_badpixels(master_flat_corr, master_illumination_model, badpixelmap_combined)
master_flat_corr[outside_traces] = np.nan

dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/badpixel_maps')
fpath = os.path.join(dirout, 'badpixelmap_master_flat')
fig, ax = plt.subplots(figsize=(7.5, 7.5))
plot_image(badpixelmap, ax=ax)
ax.set_title('Badpixel map (master flat)', size=15)
plt.savefig(fpath+'.png', dpi=300)
fits.writeto(fpath+'.fits', badpixelmap, output_verify="ignore", overwrite=True)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
plot_image(badpixelmap_combined, ax=ax)
ax.set_title('Bad pixel map (master flat + master dark)', size=15)
plt.close()


# Plot final fringe and badpixel corrected master flat.
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/master')
if not os.path.exists(dirout):
    os.mkdir(dirout)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax = plot_image(img=master_flat_corr, ax=ax, vmin=1., vmax=1e4)
plot_traces(master_flat_corr, master_flat_traces.coefs_all, ax=ax, lw=0.5)
figpath = os.path.join(dirout, 'master_flat')
ax.set_title('Master flat', size=15)
plt.tight_layout()
plt.savefig(figpath+'.png', dpi=250)

fpath = os.path.join(dirout, 'master_flat')
fits.writeto(fpath+'.fits', master_flat_corr, output_verify="ignore", overwrite=True)
fpath = os.path.join(dirout, 'trace_master_flat')
master_flat_traces.save(fpath+'.pkl')
plt.close()
