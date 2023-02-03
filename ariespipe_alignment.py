#!/usr/bin/env python
# coding: utf-8

# # Post-processing: aligment of the spectra (WASP-33)
# ---

# <b>Modules and packages

# In[11]:

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import shutil

from astropy.io import fits
import sys
ARIES_BASE_DIR = '../..'
sys.path.append(ARIES_BASE_DIR)

from aries.cleanspec import SpectralCube
from aries.cleanspec import SpectralOrder

from aries.preprocessing import plot_image
from aries.constants import TABLEAU20


# <b>Target info

# In[12]:


# Print to terminal
print('-'*50)
print('6. Align spectra in spectral time series')
print('-'*50)


# Get target info from header arguments
parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdate', type=str)
parser.add_argument('-reference_order', type=int, default=0)

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate
reference_order = args.reference_order

OBSERVATION_BASE_DIR = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}')


# <b>Algorithm parameters

# In[13]:


ALIGNMENT_MASK_DX = 15 # width around lines used for alignment mask
ALIGNMENT_TEMPLATE_MODE = 'median'
ALIGNMENT_OSR = 2 # oversampling used to interpolate between integer shifts
ALIGNMENT_SHIFT_MAX = 5 # pixel maximal shift


# ## Load badpixel corrected spectral orders

# In[14]:


dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/badcorr')
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/alignment')

if not os.path.exists(dirout):
    os.mkdir(dirout)
fpath = os.path.join(dirin, f'spectralcube_{targetname}_badcorr_result.fits')

sc = SpectralCube.load(fpath)
sc.plot(vmin=0)
plt.close()


# ## Estimates from previous wavelength calibration

# In[18]:


fpath = OBSERVATION_BASE_DIR+f'/meta/{targetname}_alignment.txt'
if not os.path.exists(fpath):
    print('\tNo configuration file found, using default configuration file.')
    config_fpath = os.path.abspath('../../config/aries_alignment.txt')
    shutil.copyfile(config_fpath, fpath)
else:
    print('\tConfiguration file found!')


# ## Align all spectral orders

# In[16]:


dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/alignment')
if not os.path.exists(dirout):
    os.mkdir(dirout)

fpath_lines = os.path.abspath(OBSERVATION_BASE_DIR+F'/meta/{targetname}_alignment.txt')
data_lines = np.loadtxt(fpath_lines, delimiter=' ', dtype=str)

orders = np.arange(1, sc.norders+1)
for n, order in enumerate(orders,1):
    so = sc.get_spectralorder(norder=order)
    so.norder += reference_order

    # create mask
    lines = np.array(data_lines[so.norder-1, 1].split(','), dtype=int)
    mask_alignment = np.zeros(so.data.shape) + 1.
    for x in lines:
        mask_alignment[:, x-ALIGNMENT_MASK_DX:x+ALIGNMENT_MASK_DX] = 0.
        
    dirout_order = os.path.join(dirout, f'order_{so.norder}')
    if not os.path.exists(dirout_order):
        os.mkdir(dirout_order)
    
    print('Aligning spectral order {}/{}'.format(n, len(orders)))
    
    # align order
    drift, data_a = so.align(mask_alignment, dirout=dirout_order,
                             template_mode=ALIGNMENT_TEMPLATE_MODE, osr=ALIGNMENT_OSR, shiftmax=ALIGNMENT_SHIFT_MAX)

    # fix mask
    t = np.where(so.mask[0,:] == 0)[0]
    imin, imax = t.min(), t.max()
    new_mask = np.ones(so.mask.shape)
    new_mask[:, imin:imax] = 0
    so.mask = new_mask
    
    so.plot(so.data)
    fname = dirout_order+f'/{targetname}_order_before_alignment'
    plt.savefig(fname+'.png', dpi=150)
    fits.writeto(fname+'.fits', so.data, overwrite=True)
    fname = dirout_order+f'/{targetname}_order_mask'
    fits.writeto(fname+'.fits', so.mask, overwrite=True)
    plt.close()
    
    data_n = so.data / so.data_normalized()
    so.plot(data_n)
    fname = dirout_order+f'/{targetname}_order_before_alignment_normalized'
    plt.savefig(fname+'.png', dpi=150)
    fits.writeto(fname+'.fits', data_n, overwrite=True)
    plt.close()
    
    res = data_n - so.data_column_mean_subtracted(data=data_n)
    so.plot(res, cmap='bwr')
    fname = dirout_order+f'/{targetname}_order_before_alignment_residual'
    plt.savefig(fname+'.png', dpi=150)
    fits.writeto(fname+'.fits', res, overwrite=True)
    plt.close()
    
    so.plot(data_a)
    fname = dirout_order+f'/{targetname}_order_aligned'
    plt.savefig(fname+'.png', dpi=150)
    fits.writeto(fname+'.fits', data_a, overwrite=True)
    plt.close()
    
    data_n = data_a / so.data_normalized(data_a)
    so.plot(data_n)
    fname = dirout_order+f'/{targetname}_order_aligned_normalized'
    plt.savefig(fname+'.png', dpi=150)
    fits.writeto(fname+'.fits', data_n, overwrite=True)
    plt.close()
    
    res = data_n - so.data_column_mean_subtracted(data=data_n)
    so.plot(res, cmap='bwr')
    fname = dirout_order+f'/{targetname}_order_aligned_residual'
    plt.savefig(fname+'.png', dpi=150)
    fits.writeto(fname+'.fits', res, overwrite=True)
    plt.close()

