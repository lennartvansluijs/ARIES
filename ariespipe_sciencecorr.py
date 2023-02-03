#!/usr/bin/env python
# coding: utf-8

# Pre-processing: science frame correction

import argparse
import os
import pickle
import sys
from settings import ARIES_BASE_DIR, DATA_BASE_DIR


sys.path.append(ARIES_BASE_DIR)
PYTHON_VERSION = sys.version_info[0]
import scipy
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec

import numpy as np
from astropy.io import fits

from aries.constants import get_tableau20_colors, TABLEAU20, ARIES_BADCOLUMNS

from aries.preprocessing import get_imgs_in_dir, get_keyword_from_headers_in_dir
from aries.preprocessing import plot_image
from aries.preprocessing import correct_badcolumns, fix_badpixels_science
from aries.preprocessing import make_badpixelmap
from aries.preprocessing import get_fits_fnames, load_imgs

from aries.traces import load_traces
from aries.traces import EchelleTraces, EchelleImageTransformer
from aries.traces import plot_traces

matplotlib.cm.Greys_r.set_bad(color='black')

# Print to terminal
print('-'*50)
print('3.c Flat correction on science frames')
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
SCIENCE_KEYWORD = 'science'
MASTER_DARK_KEYWORD = 'master_dark'


# Create science frame badpixelmap
dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr')
dirin_traces = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces')

sciences_fnames = get_fits_fnames(dirin, key='science')
sciences_traces_fnames = ['trace_'+fname for fname in sciences_fnames]

sciences = load_imgs(dirin, sciences_fnames)
sciences_traces = load_traces(dirin_traces, sciences_traces_fnames)

dirin_badpixelmaps = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/badpixel_maps')

fpath = os.path.join(dirin_badpixelmaps, 'badpixelmap_master_flat.fits')
badpixelmap_flats = fits.getdata(fpath)


exptime = get_keyword_from_headers_in_dir('EXPTIME', dirin, key=SCIENCE_KEYWORD)[0]
dirin_badpixelmap = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/badpixel_maps/')

fname = f'badpixelmap_master_dark_{int(exptime)}s.fits'
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


badpixelmap = np.array(np.logical_or(badpixelmap_darks, badpixelmap_flats), dtype=int)

fig, ax = plt.subplots(figsize=(7.5,7.5))
ax = plot_image(badpixelmap, ax=ax)
ax.set_title('Badpixelmap (flats+darks)', size=15)
fpath = os.path.join(dirin_badpixelmap, 'badpixelmap_sciences')
plt.close()
plt.savefig(fpath+'.png', dpi=250)
fits.writeto(fpath+'.fits', badpixelmap, output_verify="ignore", overwrite=True)


# Correct all sciences
dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr')
dirin_traces = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces')

sciences_fnames = get_fits_fnames(dirin, key='science')
sciences_traces_fnames = ['trace_'+fname for fname in sciences_fnames]

sciences = load_imgs(dirin, sciences_fnames)
sciences_traces = load_traces(dirin_traces, sciences_traces_fnames)


fpath = os.path.join(OBSERVATION_BASE_DIR+'/processed/badpixel_maps/badpixelmap_sciences.fits')
badpixelmap = fits.getdata(fpath)


dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr/master')
fname = 'master_flat.fits'
fpath = os.path.join(dirin, fname)
master_flat = fits.getdata(fpath)

fname = 'trace_master_flat.pkl'
fpath = os.path.join(dirin, fname)
master_flat_traces = EchelleTraces.load(fpath)


dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/flatcorr')

fontsize = 21
ticks_labelsize = 18
figsize=  (7.5,7.5)
stamp = ((400, 600), (800, 1000))
vmax = 1.5e3 # may want to make this target specific

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

print('\n\tDivision by master flat for all science frames...')
for nframe, (fname, img, traces) in enumerate(zip(sciences_fnames, sciences, sciences_traces), 1):
    # Badpixel correction
    img_corr = correct_badcolumns(img, ARIES_BADCOLUMNS)
    img_corr = fix_badpixels_science(img_corr, badpixelmap)
    
    # Flat division
    img_flatcorr = img_corr/master_flat
    img_flatcorr[np.isnan(img_flatcorr)] = 0.
    
    # finally correct all remaining zero values
    negatives_map = np.array((img_flatcorr < 0))
    img_flatcorr = fix_badpixels_science(img_flatcorr, negatives_map)
    
    # Save output
    fpath = os.path.join(dirout, fname)
    fits.writeto(fpath+'.fits', img_flatcorr, output_verify="ignore", overwrite=True)
    
    # Create and save overview plot
    fpath = os.path.join(dirout, 'plot_'+fname)
    fig, ax = plt.subplots(figsize=(7.5,7.5))
    
    cmap = plt.get_cmap('Greys_r')
    cmap.set_bad('grey',1.)
    
    ax, cbar = plot_image(img_flatcorr, ax=ax, vmin=0, vmax=1.68, return_cbar=True)
    
    plot_traces(img_flatcorr, master_flat_traces.coefs_all, ax=ax, lw=2.5, colors='blue', yoffset=0, label='master flat trace')
    #plot_traces(img_flatcorr, master_flat_traces.coefs_all, ax=ax, lw=2.5, colors='blue', yoffset=-5)
    plot_traces(img_flatcorr, traces.coefs_all, colors='red', lw=2.5, ax=ax, label='science trace')
    ax.set_title('{}.fits'.format(fname), fontsize=fontsize)
    ax.set_xlim(stamp[0][0],stamp[0][1])
    ax.set_ylim(stamp[1][0],stamp[1][1])
    
    ax.set_xlabel('', fontsize=fontsize)
    ax.set_ylabel('', fontsize=fontsize)
    cbar.remove()
    
    legend_without_duplicate_labels(ax)
    
    plt.tight_layout()
    plt.savefig(fpath+'.png', dpi=100)
    plt.close()
print('Done.')
