#!/usr/bin/env python
# coding: utf-8


# Pre-processing: dark correction


import argparse
import matplotlib
import os
import sys
import warnings


import numpy as np
from matplotlib import pyplot as plt


from astropy.io import fits
from settings import ARIES_BASE_DIR, DATA_BASE_DIR
sys.path.append(ARIES_BASE_DIR)

from aries.preprocessing import get_master, correct_for_dark_current, plot_image
from aries.utils import save_imgs_as_fits
from aries.preprocessing import fix_badpixels
from aries.preprocessing import get_master, get_imgs_in_dir, get_keyword_from_headers_in_dir
from aries.preprocessing import make_badpixelmap
from aries.preprocessing import identify_badcolumns, correct_badcolumns
from aries.preprocessing import make_crosshair_footing, make_point_footing
from aries.preprocessing import is_flat, is_science


# Print to terminal
print('-'*50)
print('2. Dark current correction')
print('-'*50)


# Get target info from header arguments
parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdate', type=str)

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate
OBSERVATION_BASE_DIR = os.path.abspath(DATA_BASE_DIR + f'/{targetname}/{obsdate}')



# Dark correction parameters
FOOTING_SIZE = 11 # pixel
FOOTING_MODE = 'crosshair'
MASTER_DARK_KEYWORD = 'master_dark'
DARK_KEYWORD = 'dark'
SCIENCE_KEYWORD = 'science'
FLAT_KEYWORD = 'flat'
MASTER_METHOD = 'median'
BADCOLS_SIGMA = 3.


# Now let's create the master darks. Master darks are made for each unique observing time (for example a 'short' (60s) and a 'long' (300s) master dark'. We correct for badpixels and create badpixel map. Hot are correct from their neighbours median using a specified footing pattern.

# Create footing pattern
footing = make_crosshair_footing(size=FOOTING_SIZE)



# Load all files and specifiy new output directories.
parentdirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr/')
if not os.path.exists(parentdirout):
    os.mkdir(parentdirout)

dirout_master = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr/master')
if not os.path.exists(dirout_master):
    os.mkdir(dirout_master)
    
dirout_bads = OBSERVATION_BASE_DIR+'/processed/badpixel_maps'
if not os.path.exists(dirout_bads):
    os.mkdir(dirout_bads)

darks_dirname = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/corquad')
darks = get_imgs_in_dir(darks_dirname, key=DARK_KEYWORD)
exptime_darks = get_keyword_from_headers_in_dir('EXPTIME', darks_dirname, key=DARK_KEYWORD)


# Seperate in long and short exposure darks, if relevant
exptimes = np.unique(exptime_darks)
nexptimes = exptimes.size

print('\n\tCreating master darks...')
for exptime in np.unique(exptime_darks):
    # create master dark
    darks_s = darks[np.where(exptime_darks == exptime)]
    master_dark = get_master(darks_s, method=MASTER_METHOD)
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax = plot_image(img=master_dark, ax=ax, vmin=0, vmax=200)
    figpath = os.path.join(dirout_master, f'masterdark_{int(exptime)}s')
    ax.set_title(f'Master dark (texp={int(exptime)}s)', size=15)
    plt.savefig(figpath+'.png', dpi=250)
    plt.close()
    
    # correct bads
    badcolumns = identify_badcolumns(master_dark, sigma=BADCOLS_SIGMA)
    master_dark_bccorr = correct_badcolumns(master_dark, badcolumns=badcolumns)
    badpixelmap = make_badpixelmap(master_dark_bccorr, sigma=BADCOLS_SIGMA)
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    plot_image(badpixelmap, ax=ax)
    figpath = os.path.join(dirout_bads, f'badpixelmap_master_dark_{int(exptime)}s')
    
    fpath = os.path.join(dirout_bads, f'badpixelmap_master_dark_{int(exptime)}s.fits')
    fits.writeto(fpath, badpixelmap, overwrite=True)
    
    master_dark_corr = fix_badpixels(master_dark_bccorr, badpixelmap,
                                     size=FOOTING_SIZE, mode=FOOTING_MODE)
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax = plot_image(img=master_dark_corr, ax=ax, vmin=0, vmax=200)
    figpath = os.path.join(dirout_master, f'masterdark_{int(exptime)}s')
    ax.set_title(f'Master dark after badpixel correction (texp={int(exptime)}s)', size=15)
    plt.savefig(figpath+'.png', dpi=250)
    plt.close()
    
    hdr = fits.Header()
    hdr['EXPTIME'] = exptime
    hdr['OBJECT'] = MASTER_DARK_KEYWORD
    fpath = os.path.join(dirout_master, f'master_dark_{int(exptime)}s.fits')
    fits.writeto(fpath, master_dark, header=hdr, overwrite=True)
    print('\t'f'Created {exptime} s Master dark.')
print('Done.')


# Dark correction on sciences/flats
dirin = OBSERVATION_BASE_DIR+'/processed/corquad'
dirout = OBSERVATION_BASE_DIR+'/processed/darkcorr'
if not os.path.exists(dirout):
    os.mkdir(dirout)

fnames = [fname for fname in os.listdir(dirin) if           (is_flat(fname) or is_science(fname))]

print('\n\tDoing dark correction on all images...')
for fname in fnames:
    print('\t{}'.format(fname))
    # Load data
    fpath = os.path.join(dirin, fname)
    img = np.array(fits.getdata(fpath, ignore_missing_end=True))
    header = fits.getheader(fpath, ignore_missing_end=True)
    exptime = header['EXPTIME']
    
    # Try to load matching dark
    try:
        fpath = os.path.join(dirout_master, f'master_dark_{int(exptime)}s.fits')
        master_dark = fits.getdata(fpath)
        sf = 1.
    except FileNotFoundError:
        fpaths = [f for f in os.listdir(dirout_master) if (MASTER_DARK_KEYWORD in f) and f.endswith('.fits')]
        nmaster_darks = len(fpaths)
        if nmaster_darks > 0:
            reference_fpath = os.path.join(os.path.join(dirout_master, fpaths[0]))
            reference_master_dark = fits.getdata(reference_fpath)
            hdr = fits.getheader(reference_fpath, ignore_missing_end=True)
            exptime_master_dark = float(hdr['EXPTIME'])
            sf = (exptime/exptime_master_dark)
            warnings.warn(f'No matching master dark of texp = {int(exptime)}s for file {fname}'                           f'Using linear scaling using master dark of texp = {int(exptime_master_dark)}s.')
        else:
            raise FileNotFoundError('No master darks found.')

    # correct
    img = img - sf * master_dark
        
    # Write to fits file
    fpathout = os.path.join(dirout, fname)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits.writeto(fpathout, img, header,
                     output_verify="ignore", overwrite=True)
print('Done.')

#!/usr/bin/env python
# coding: utf-8

