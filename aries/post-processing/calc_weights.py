#!/usr/bin/env python
# coding: utf-8

# ## WASP-33 determine cross correlation weighting

# I determine the cross-correlation weighthing from log L maps.
# First I run log L with the best model as for the real data at alpha=1.
# 
# Secondly I run log L grid with an injected model at alpha=5.
# Alpha=5  is chosen strong enough to detect clearly,
# but not too strong such that the signal affects the noise minimally.

# In[3]:

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import sys

from matplotlib import gridspec
from itertools import product

ARIES_BASE_DIR = os.path.abspath('../..')
sys.path.append(ARIES_BASE_DIR)

from aries.constants import TABLEAU20, ARIES_NORDERS
from aries.crosscorrelate import shift_to_new_restframe
from aries.systemparams import systems, targetname_to_key


# In[4]:


NO_TXT_EXTENSION = slice(0,-4,1)


# <b> From the cross-correlation matrices

# In[5]:

# Get target info from header arguments
parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdates', type=str)
parser.add_argument('-dirname_models', type=str)
parser.add_argument('-fname_model', type=str)
parser.add_argument('-dirname_residuals', type=str)
parser.add_argument('-apply_hpf_to_model', type=bool)
parser.add_argument('-orders', type=str)
parser.add_argument('-ninj', type=int)


args = parser.parse_args()
targetname = args.targetname
obsdates = args.obsdates.split(' ')
dirname_models = args.dirname_models
model_fname = args.fname_model
detrending_method = args.dirname_residuals
apply_hpf_to_model = args.apply_hpf_to_model
orders = np.array(args.orders.split(' '), dtype=int)
ninj = args.ninj

systemparams= systems[targetname_to_key(targetname)]


if apply_hpf_to_model:
    model_extension = '_hpf'
else:
    model_extension = ''



# Cross-correlation
RV_MIN = -500e3 # m/s
RV_MAX = 500e3 # m/s
DELTA_RV = 5e3 
nshifts = int((RV_MAX-RV_MIN)/DELTA_RV + 1)
rv_sample = np.linspace(RV_MIN, RV_MAX, nshifts)

for obsdate in obsdates:
    print('\n\tLoading barycentric data...')
    dirin_data =  ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}/processed/injected/{dirname_models}/{model_fname[NO_TXT_EXTENSION]}{model_extension}/x0/significances/{detrending_method}/crosscorr'
    dirin_inj = ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}/processed/injected/{dirname_models}/{model_fname[NO_TXT_EXTENSION]}{model_extension}/x{ninj}/significances/{detrending_method}/crosscorr'
    dirin_meta = f'/home/lennart/measure/data/{targetname}/{obsdate}/meta'
    fpath = os.path.join(dirin_meta, 'science_times.txt')
    times_bjdtbd = np.loadtxt(fpath, dtype='str', delimiter=',').T[4].astype('float')

    fpath = os.path.join(dirin_meta, 'vbarycorr.txt')
    phase, vbary, rvplanet = np.loadtxt(fpath)
    
    vsys = systemparams['vsys']
    kp = systemparams['kp']
    print('Done.')
    
    plt.figure(figsize=(10,10))
    NROWS, NCOLS = (6, 5)
    fig = plt.figure(figsize=(15,15))
    gs = gridspec.GridSpec(nrows=NROWS,ncols=NCOLS)

    ccmatrix_data_1d_list = []
    ccmatrix_inj_1d_list = []
    ccmatrix_diff_1d_list = []
    print('\n\tCalculating weights from injected data...')
    for i, order in enumerate(orders):
        # load data cross-correaltion matrix
        f = os.path.join(dirin_data, f'{targetname}_ccmatrix_order_{order}_{obsdate}.fits')
        ccmatrix_data = np.array(fits.getdata(f))

        # load injected cross-correlation matrix
        f_inj = os.path.join(dirin_inj, f'{targetname}_ccmatrix_order_{order}_{obsdate}.fits')
        ccmatrix_inj = np.array(fits.getdata(f_inj))

        # shift to injected rest frame
        rv = -vbary + vsys + kp * np.sin(phase*2*np.pi)
        ccmatrix_data_s = shift_to_new_restframe(ccmatrix_data, rv0=rv_sample, rvshift=rvplanet)
        ccmatrix_inj_s = shift_to_new_restframe(ccmatrix_inj, rv0=rv_sample, rvshift=rvplanet)
        ccmatrix_diff_s = ccmatrix_inj_s - ccmatrix_data_s

        # sum over all frames
        ccmatrix_data_1d = np.nansum(ccmatrix_data_s, axis=0)
        ccmatrix_inj_1d = np.nansum(ccmatrix_inj_s, axis=0)
        ccmatrix_diff_1d = ccmatrix_inj_1d - ccmatrix_data_1d

        # append to list
        ccmatrix_data_1d_list.append(ccmatrix_data_1d)
        ccmatrix_inj_1d_list.append(ccmatrix_inj_1d)
        ccmatrix_diff_1d_list.append(ccmatrix_diff_1d)
    
    gsloc = [loc for loc in product(range(NROWS), range(NCOLS))][:len(orders)]

    peakheight_values = np.zeros(ARIES_NORDERS)
    for i, (order, loc) in enumerate(zip(orders, gsloc)):
        ax = fig.add_subplot(gs[loc[0]:loc[0]+1, loc[1]:loc[1]+1])
        ax.set_title(f'Order {order}', size=10)
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.plot(ccmatrix_data_1d_list[i], color='k')
        #ax.plot(ccmatrix_inj_1d_list[i], color='r')
        ax.plot(ccmatrix_diff_1d_list[i], color='b')
        ax.set_ylim(-1,1)
        peakheight_values[order-1] = np.max(ccmatrix_diff_1d_list[i] - np.min(ccmatrix_diff_1d_list[i]))
    #     fpathout = os.path.join(dirout_weights, f'{obsdate}_{model_fname[NO_TXT_EXTENSION]}_plot_ccmatrix_diff_1d_all_orders.png')
    #     plt.savefig(fpathout)
    #     plt.show()

    cc_weights = peakheight_values/np.nansum(peakheight_values)
    print('Done.')


    dirout_weights = ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}/processed/weights/{dirname_models}'
    if not os.path.exists(dirout_weights):
        os.makedirs(dirout_weights)
    fpathout = os.path.join(dirout_weights, f'{obsdate}_{model_fname[NO_TXT_EXTENSION]}{model_extension}_cc_weights.txt')
    data_out = np.array([np.arange(1, ARIES_NORDERS+1), cc_weights])
    np.savetxt(fpathout, data_out, header='1 order, 2 cc weights')


    # fpathout = os.path.join(dirout, 'ccmatrix_diff_1d_list.npy')
    #np.save(fpathout, np.array(ccmatrix_diff_1d_list))


    data = np.loadtxt(fpathout)
    cc_weights = np.array([w for norder, w in zip(data[0,:], data[1,:]) if norder in orders])
    plt.plot(cc_weights)
    fpathout = os.path.join(dirout_weights, f'{obsdate}_{model_fname[NO_TXT_EXTENSION]}{model_extension}_plot_cc_weights.png')
    plt.savefig(fpathout)
    plt.close()


# In[11]:


# parentdir = '/home/lennart/measure/data/{targetname}/20161015/processed/'
# f = 'w33_aries_josh_selfconsistent/pca_7iter_masked_hpf/WASP-33.redist=0.5.pp.z10.more.hires.7/bl19_gridsearch/loglike_all_orders.fits'
# fpath_data = os.path.join(parentdir, f)
# logL_all_orders = fits.getdata(fpath_data)
# f_inj = 'injected(2)/WASP-33.redist=0.5.pp.z10.more.hires.7.csv_x5/significances/pca_7iter_masked_hpf/bl19_gridsearch/loglike_all_orders.fits'
# fpath_data_inj = os.path.join(parentdir, f_inj)
# logL_all_orders_inj = fits.getdata(fpath_data_inj)
# dlogL_all_orders = logL_all_orders_inj - logL_all_orders

# badorders = [7,8,9,11]
# orders = [n for n in range(1, 1+ARIES_NORDERS) if n not in badorders]


# for i, order in enumerate(orders):
#     #plt.imshow(logL_all_orders[:,:,i])
#     #plt.colorbar()
#     #plt.show()

#     #plt.imshow(logL_all_orders_inj[:,:,i])
#     #plt.colorbar()
#     #plt.show()

#     #nx, ny, norders = dlogL_all_orders.shape
#     #print(dlogL_all_orders[int(ny/2),int(nx/2),i])
#     plt.imshow(dlogL_all_orders[:,:,i])
#     plt.colorbar()
#     plt.show()
    


# In[5]:


# weights_fpath = '/home/lennart/measure/data/{targetname}/20161015/processed/injected/' \
# 'WASP-33.redist=0.5.pp.z10.more.hires.7.csv_x5/significances/pca_7iter_masked_hpf/crosscorr/cc_weights.txt'
# data = np.loadtxt(weights_fpath)
# cc_weights = np.array([w for norder, w in zip(data[0,:], data[1,:]) if norder in orders])


# In[6]:


# ind_sorted = np.argsort(cc_weights)
# cc_weights_cumulative = np.cumsum(cc_weights[ind_sorted])[::-1]
# plt.title('Cumulative weight per order (sorted)', size=12)
# plt.bar(x=np.arange(len(cc_weights_cumulative)),
#                     height=cc_weights_cumulative)

# order_cutoff = 5
# print(f'The best {order_cutoff} orders include {100-np.round(cc_weights_cumulative[order_cutoff-1]*100,1)}% of the weights.')


# In[ ]:





# In[ ]:





# In[ ]:




