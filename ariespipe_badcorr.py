#!/usr/bin/env python
# coding: utf-8

# # Post-processing: badpixel correction (WASP-33)
# ---

# <b>Modules and packages

# In[1]:

import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from astropy.io import fits
import sys
from settings import ARIES_BASE_DIR, DATA_BASE_DIR
sys.path.append(ARIES_BASE_DIR)

from aries.cleanspec import SpectralCube
from aries.preprocessing import plot_image
from aries.constants import TABLEAU20


# <b>Target info

# In[2]:


# Get target info from header arguments
parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdate', type=str)

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate

OBSERVATION_BASE_DIR = os.path.abspath(DATA_BASE_DIR + f'/{targetname}/{obsdate}')


# <b>Algorithm parameters

# In[ ]:





# ## Load Spectral Time Series 

# First let's load the spectral time series for all spectral orders.

# In[3]:


# dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra')
# fpath = os.path.join(dirin, f'spectralcube_{targetname}_raw.fits')

# spectralcube = SpectralCube.load(fpath)
# spectralcube.apply_mask(apply_to_value=0)
# axes = spectralcube.plot(vmin=0, figtitle='Spectral Time Series All Orders')
# #plt.savefig(f'{targetname}_rawspectra.pdf')
# plt.show()

# normalized_rows = spectralcube.normalized_rows()
# spectralcube.data /= normalized_rows
# spectralcube.save(os.path.join(dirin, f'spectralcube_{targetname}_normalized_rows.fits'))
# axes = spectralcube.plot(vmin=0.5, vmax=1.5, figtitle='Normalized Spectral Time Series All Orders')
# plt.show()

# mean_columns = spectralcube.mean_columns()
# spectralcube.data -= mean_columns
# spectralcube.save(os.path.join(dirin, f'spectralcube_{targetname}_normalized_rows_mean_columns_subtracted.fits'))
# axes = spectralcube.plot(cmap='bwr', vmin=-0.1, vmax=0.1, figtitle='Residual Spectral Time Series All Orders')
# plt.show()
# plt.close()


# ## Iteratively perform badpixel correction

# In[4]:


dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra')
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/badcorr')
if not os.path.exists(dirout):
    os.mkdir(dirout)

fpath = os.path.join(dirin, f'spectralcube_{targetname}_raw.fits')
sc = SpectralCube.load(fpath)

# orders = np.arange(1, sc.norders+1)
# for n, order in enumerate(orders,1):
#     so = sc.get_spectralorder(norder=order)
#     so.plot()
#     plt.show()


# In[5]:


dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra')
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/badcorr')
if not os.path.exists(dirout):
    os.mkdir(dirout)

fpath = os.path.join(dirin, f'spectralcube_{targetname}_raw.fits')
spectralcube = SpectralCube.load(fpath)
spectralcube.apply_mask(apply_to_value=0)

detection_threshold = 4.
niter = 5
for n in range(1, niter+1):
    print('Badcorr run {}/{}'.format(n, niter))
    dirout_run = os.path.join(dirout, 'run'+str(n))
    if not os.path.exists(dirout_run):
        os.mkdir(dirout_run)
    
    spectralcube.plot(vmin=0, figtitle='Before: Bad pixel/column corrected')
    plt.savefig(os.path.join(dirout_run, 'sc_before_badcorr.png'), dpi=150)
    spectralcube.save(os.path.join(dirout_run, f'spectralcube_{targetname}_before_badcorr.fits'))
    plt.close()
    
    data_c = np.copy(spectralcube.data)
    normalized_rows = spectralcube.normalized_rows()
    spectralcube.data /= normalized_rows
    mean_columns = spectralcube.mean_columns()
    spectralcube.data -= mean_columns

    blobmap, gl, sigmamap = spectralcube.detect_blobs(return_full=True)

    fits.writeto(os.path.join(dirout_run, f'spectralcube_{targetname}_gl.fits'), gl, overwrite=True)
    fits.writeto(os.path.join(dirout_run, f'spectralcube_{targetname}_blobmap.fits'), np.array(blobmap, dtype=int), overwrite=True)
    fits.writeto(os.path.join(dirout_run, f'spectralcube_{targetname}_sigmamap.fits'), sigmamap, overwrite=True)
    
    badpixelmap = np.array(sigmamap > detection_threshold, dtype=bool)
    fits.writeto(os.path.join(dirout_run, f'spectralcube_{targetname}_badpixelmap.fits'), np.array(badpixelmap, dtype=int), overwrite=True)
    
    nbads = np.sum(badpixelmap)
    print('Run {}: found {} bad pixels.'.format(n, nbads))
    if nbads == 0:
        break
        
    spectralcube.data = data_c
    spectralcube.correct_badpixels(badpixelmap)
    spectralcube.correct_badcolumns(medfilt_size=3, sigma=5.)
    
    spectralcube.plot(vmin=0, figtitle='Bad pixel/column corrected')
    plt.savefig(os.path.join(dirout_run, 'sc_badcorr.png'), dpi=150)
    spectralcube.save(os.path.join(dirout_run, f'spectralcube_{targetname}_badcorr.fits'))
    plt.close()
    
    data_badcorr_c = np.copy(spectralcube.data)
    normalized_rows = spectralcube.normalized_rows()
    spectralcube.data /= normalized_rows
    spectralcube.plot(vmin=0, figtitle='Normalized Badcorr')
    plt.savefig(os.path.join(dirout_run, 'sc_normalized_badcorr.png'), dpi=150)
    spectralcube.save(os.path.join(dirout_run, f'spectralcube_{targetname}_badcorr_normalized_rows.fits'))
    plt.close()
    
    mean_columns = spectralcube.mean_columns()
    spectralcube.data -= mean_columns
    spectralcube.plot(cmap='bwr', figtitle='Residual Badcorr')
    plt.savefig(os.path.join(dirout_run, 'sc_residual_badcorr.png'))
    spectralcube.save(os.path.join(dirout_run, f'spectralcube_{targetname}_badcorr_normalized_rows_mean_columns_subtracted.fits'))
    plt.close()
    
    spectralcube.data = data_badcorr_c


# ## Bad column correction
# Let's run one bad column correction before moving on.

# In[6]:


fpath = os.path.join(dirout_run, f'spectralcube_{targetname}_badcorr.fits')
spectralcube = SpectralCube.load(fpath)
spectralcube.correct_badcolumns(medfilt_size=3, sigma=5.)
spectralcube.save(os.path.join(dirout, f'spectralcube_{targetname}_badcorr_result.fits'))

normalized_rows = spectralcube.normalized_rows()
spectralcube.data /= normalized_rows
spectralcube.save(os.path.join(dirout, f'spectralcube_{targetname}_badcorr_result_normalized_rows.fits'))


# ---

# ## Badpixelmap using Blob Detection Algorithm

# In[7]:


# blobmap, gl, sigmamap = spectralcube.detect_blobs(return_full=True)

# fits.writeto(f'spectralcube_{targetname}_gl.fits', gl, overwrite=True)
# fits.writeto(f'spectralcube_{targetname}_blobmap.fits', np.array(blobmap, dtype=int), overwrite=True)
# fits.writeto(f'spectralcube_{targetname}_sigmamap.fits', sigmamap, overwrite=True)


# In[8]:


# sigmamap = fits.getdata(f'spectralcube_{targetname}_sigmamap.fits')
# badpixelmap = np.array(sigmamap > 4, dtype=bool)
# fits.writeto(f'spectralcube_{targetname}_badpixelmap.fits', np.array(badpixelmap, dtype=int), overwrite=True)

# axes = spectralcube.plot(cmap='Reds_r', vmin=0, vmax=4, data=sigmamap, figtitle='Bad Pixel Sigma Map')
# axes = spectralcube.plot(cmap='bwr', vmin=-0.1, vmax=0.1, data=gl, figtitle='Gaussian Laplace of Residual')


# Plot identified badpixels.

# In[9]:


# spectralcube = SpectralCube.load(f'spectralcube_{targetname}_normalized_rows.fits')
# for norder in range(1, spectralcube.norders+1):
#     ax, cax = spectralcube.plot_spectral_order(norder,
#                          figtitle='Before Badcorr Normalized Spectral Time Series All Orders',
#                          origin='bottom', vmin=0, vmax=2, cmap='hot')
#     bads = np.where(badpixelmap[norder-1,:,:])[:]
#     ax.scatter(bads[1]+1, bads[0]+1, s=150, edgecolor='w', facecolor='none', alpha=1)
#     ax.set_xlim(1, spectralcube.npixels)
#     ax.set_ylim(1, spectralcube.nobs)
#     plt.show()


# ## Apply bad pixel map + Bad column correction

# In[10]:


# spectralcube = SpectralCube.load(os.path.join(dirin, f'spectralcube_{targetname}_raw.fits'))
# spectralcube.apply_mask(apply_to_value=0)
# spectralcube.correct_badpixels(badpixelmap)
# spectralcube.correct_badcolumns(medfilt_size=3, sigma=5.)
# axes = spectralcube.plot(vmin=0, figtitle='Bad pixel/column corrected')

# normalized_rows = spectralcube.normalized_rows()
# spectralcube.data /= normalized_rows
# spectralcube.save(f'spectralcube_{targetname}_badcorr_normalized_rows.fits')
# axes = spectralcube.plot(vmin=0, figtitle='Normalized Badcorr')

# mean_columns = spectralcube.mean_columns()
# spectralcube.data -= mean_columns
# spectralcube.save('spectralcube_{targetname}_badcorr_normalized_rows_mean_columns_subtracted.fits')
# axes = spectralcube.plot(cmap='bwr', figtitle='Residual Badcorr')


# In[ ]:




