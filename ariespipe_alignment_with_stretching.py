#!/usr/bin/env python
# coding: utf-8

# # Post-processing: aligment of the spectra (WASP-33)
# ---
# Based on Brogi et al. 2016.
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from itertools import product

from astropy.io import fits
import sys
from settings import ARIES_BASE_DIR, DATA_BASE_DIR
sys.path.append(ARIES_BASE_DIR)

from aries.cleanspec import SpectralCube
from aries.cleanspec import SpectralOrder
from aries.cleanspec import clip_mask
from aries.ipfit import correct_continuum
from aries.preprocessing import plot_image
from aries.constants import TABLEAU20


# ---
# Program header
# ---


sf, nc = 20, 30
print('\n'+'-'*(2*sf+nc))
print('-'*sf + '      MEASURE alignment      ' + '-'*sf)
print('-'*sf + '    (Brogi et al. 2016)      ' + '-'*sf)
print('-'*sf + '     by Lennart van Sluijs   ' + '-'*sf)
print('-'*(2*sf+nc))


# ---
# Parse input parameters
# ---


parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdate', type=str)

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate


OBSERVATION_BASE_DIR = os.path.abspath(DATA_BASE_DIR+ f'/{targetname}/{obsdate}')


# Algorithm parameters
ALIGNMENT_MASK_DX = 15 # width around lines used for alignment mask
ALIGNMENT_TEMPLATE_MODE = 'median'
ALIGNMENT_OSR = 2 # oversampling used to interpolate between integer shifts
ALIGNMENT_SHIFT_MAX = 5 # pixel maximal shift


# Load all spectral orders
dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/badcorr')
fpath = os.path.join(dirin, f'spectralcube_{targetname}_badcorr_result.fits')
sc = SpectralCube.load(fpath)


# Define output directory after alignment of each order
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/alignment_with_stretching')
if not os.path.exists(dirout):
    os.mkdir(dirout)


# Main
orders = np.arange(1, sc.norders+1)
for norder, order in enumerate(orders,1):
    print(f'Spectral order {norder}/{len(orders)}')
    so = sc.get_spectralorder(norder=order)
    
    # fix mask
    t = np.where(so.mask[0,:] == 0)[0]
    imin, imax = t.min(), t.max()
    new_mask = np.ones(so.mask.shape)
    new_mask[:, imin:imax] = 0
    so.mask = new_mask

    dirout_order = os.path.join(dirout, f'order_{order}')
    if not os.path.exists(dirout_order):
        os.mkdir(dirout_order)


    so.plot()
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

    # Use clipped data to find alignment solution
    data, clip = clip_mask(data_n, so.mask, return_clip=True)
    nframes, nx = data.shape
    data_pxl = so.wavsolution[0][~clip]


    dx_iter = [2,1,0.5,0.25]
    nx = 7
    dx_triplet_prev = np.zeros(shape=(nframes,3))
    print('Trying multiple alignment solutions...')
    for niter, dx in enumerate(dx_iter):
        print(f'\tIteration {niter+1}/{len(dx_iter)}')
        dx_values = np.linspace(-dx,dx,nx)
        pxl_triplet = np.array(
            [
                data_pxl[0],
                data_pxl[int(len(data_pxl)/2.)],
                data_pxl[-1]
            ]
        )
        polydeg = 2

        #     from itertools import product
        #     x_stretched_all = []
        #     p_all = []
        #     for p in product(dx_values, dx_values, dx_values):
        #         print(p_prev+p+anchors)
        #         z = np.polyfit(anchors, p_prev+p+anchors, deg=polydeg)
        #         poly = np.poly1d(z)
        #         #plt.plot(so.wavsolution[0], poly(so.wavsolution[0]))
        #         x_stretched = poly(data_pxl)
        #         x_stretched_all.append(x_stretched)
        #         p_all.append(p+p_prev)

        # check which shift is best
        spec_ref = np.array(data[0,:])
        #ntrials = len(x_stretched_all)
        dx_triplet_all = []
        for row in range(1, data.shape[0]):
            #print(f'\tCross-correlating frame {row+1}/{data.shape[0]}')
            spec = data[row,:]
            cc_values = []
            spec_stretched_all = []
            for i, dx_triplet in enumerate(product(dx_values, dx_values, dx_values)):
                pxl_offset = dx_triplet_prev[row,:] + dx_triplet
                z = np.polyfit(pxl_triplet, pxl_offset+pxl_triplet, deg=polydeg)
                poly = np.poly1d(z)
                x_stretched = poly(data_pxl)

                spec_stretched = np.interp(x=data_pxl, xp=x_stretched, fp=data[row,:])
                cc = np.corrcoef(spec_ref, spec_stretched)[0,1]
                cc_values.append(cc)
                spec_stretched_all.append(spec_stretched)
                dx_triplet_all.append(dx_triplet)

            imax = np.argmax(np.array(cc_values))
            dx_triplet_prev[row,:] += dx_triplet_all[imax]


    # Use unclipped data again for final alignment
    data = np.array(so.data)
    nframes, nx = so.data.shape
    data_pxl = so.wavsolution[0]

    # Finally align
    print('Aligning...')
    spec_shifted = np.array(data)
    for row in range(1, spec_shifted.shape[0]):
        print(f'\tAligning frame {row+1}/{data.shape[0]}')
        z = np.polyfit(pxl_triplet, dx_triplet_prev[row,:]+pxl_triplet, deg=polydeg)
        poly = np.poly1d(z)
        x_stretched_best = poly(data_pxl)
        spec_shifted[row,:] = np.interp(x=data_pxl, xp=x_stretched_best, fp=data[row,:])

    #     plt.plot(data_pxl, data[row,:], color='r', label='data')
    #     plt.plot(data_pxl, spec_shifted[row,:], color='b', label='stretched')
    #     plt.plot(data_pxl, spec_ref, color='k', label='reference', ls='--')
    #     plt.xlim(510,560)
    #     plt.legend()
    #     plt.title(f'cc={cc_values[imax]}; offset={dx_triplet_prev[row,:]}')
    #     plt.show()

    #         plt.plot(np.array(cc_values))
    #         plt.show()
            #spec_shifted[row,:] = np.interp(x=data_pxl, xp=x_stretched_all[imax], fp=spec)

    so_new = SpectralOrder(
        data=spec_shifted,
        mask=so.mask,
        norder=so.norder,
        target=targetname,
        phase=so.phase
    )


    so_new.plot(so_new.data)
    fname = dirout_order+f'/{targetname}_order_aligned'
    plt.savefig(fname+'.png', dpi=150)
    fits.writeto(fname+'.fits', so_new.data, overwrite=True)
    plt.close()

    data_n = so_new.data / so_new.data_normalized(so_new.data)
    so_new.plot(data_n)
    fname = dirout_order+f'/{targetname}_order_aligned_normalized'
    plt.savefig(fname+'.png', dpi=150)
    fits.writeto(fname+'.fits', data_n, overwrite=True)
    plt.close()

    res = data_n - so_new.data_column_mean_subtracted(data=data_n)
    so_new.plot(res, cmap='bwr')
    fname = dirout_order+f'/{targetname}_order_aligned_residual'
    plt.savefig(fname+'.png', dpi=150)
    fits.writeto(fname+'.fits', res, overwrite=True)
    plt.close()
    print('\n')
