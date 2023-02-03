#!/usr/bin/env python
# coding: utf-8

# # WASP-33 b: build template library

# Build a library of model templates for each spectral order. This avoids re-computing these template every time.

# <b> Modules and packages

# In[15]:

import argparse
import csv
import os
import sys

from astropy.io import fits
from astropy.constants import G, au, M_sun, R_sun, R_jup
import matplotlib.pyplot as plt
import numpy as np

ARIES_BASE_DIR = '../..'
sys.path.append(ARIES_BASE_DIR)
sys.path.append(os.path.abspath(ARIES_BASE_DIR)+'/lib')


from aries.cleanspec import TemplateOrder, planck, apply_highpass_filter
from aries.utils import load_planet_synthetic_spectrum
from aries.constants import TABLEAU20, ARIES_NORDERS
from aries.systemparams import systems, targetname_to_key
from scipy.stats import binned_statistic
from aries.ipfit import gaussian_convolution_kernel


# <b>Algorithm parameters

# In[16]:


# targetname = 'wasp33'
# obsdates = ['20161015','20161019','20161020']
# dirname_models = "w33_gcm_elsie"
# load_synthetic_spectrum_mode = 'gcm_elsie' #'phoenix_josh' 
# dirin_models = os.path.abspath(ARIES_BASE_DIR + f"/models/{targetname}/{dirname_models}")
# #fname_models = [fname for fname in os.listdir(dirin_models) if fname.endswith('.csv') and ('structure' not in fname)]
# fname_models = [fname for fname in os.listdir(dirin_models) if fname.endswith('.txt')]
# IP_reference_order = 23
# WAVELENGTH_PADDING = 0.2
# badorders = [7,8,9,11]
# orders = [n for n in range(1, ARIES_NORDERS+1) if n not in badorders]
# dirname_residuals = "pca_7iter_masked_hpf"
# subtract_continuum = False


# In[17]:

# Get target info from header arguments
parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdates', type=str)
parser.add_argument('-dirname_models', type=str)
parser.add_argument('-dirin_models', type=str)
parser.add_argument('-fname_models', type=str)
parser.add_argument('-dirname_residuals', type=str)
parser.add_argument('-load_synthetic_spectrum_mode', type=str)
parser.add_argument('-apply_hpf_to_model', type=bool)
parser.add_argument('-orders', type=str)
parser.add_argument('-IP_reference_order', type=int)

args = parser.parse_args()
targetname = args.targetname
obsdates = args.obsdates.split(' ')
dirname_models = args.dirname_models
dirin_models = args.dirin_models
fname_models = args.fname_models.split(' ')
dirname_residuals = args.dirname_residuals
load_synthetic_spectrum_mode = args.load_synthetic_spectrum_mode
apply_hpf_to_model = args.apply_hpf_to_model
orders = np.array(args.orders.split(' '), dtype=int)
IP_reference_order = args.IP_reference_order

HIGHPASSFILTER_FREQ_CUTOFF = 1./50.
WAVELENGTH_PADDING = 0.2

subtract_continuum = False
system = systems[targetname_to_key(targetname)]


# Overwrite models list:

# In[18]:



# In[19]:


# plt.close()
# from scipy import stats
# def apply_wilks_theorem(logL_values, degrees_of_freedom):
#     """Apply wilks theorem."""
#     dlogL_values = (logL_values - logL_values.max())
#     sigma_values = np.zeros(dlogL_values.size)
#     for i in range(dlogL_values.size):
#         p_value = stats.chi2.sf(-2*dlogL_values[i], degrees_of_freedom)
#         sigma_values[i] = stats.norm.isf(p_value/2.) # convert p-value into corresponding sigma-value
#     return sigma_values

# import matplotlib as mpl
# from matplotlib import colors

# fig, ax = plt.subplots()

# sigma_values = apply_wilks_theorem(logL_values=np.array(logL_list), degrees_of_freedom=7)
# ind_sorted = np.argsort(sigma_values)[::-1]
# theta_list = np.array(theta_list)
# norm = mpl.colors.Normalize(vmin=0, vmax=5)
# cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greys_r)
# cmap.set_array([])

# for theta, sigma in zip(theta_list[ind_sorted], sigma_values[ind_sorted]):
#     if theta[-1] == 1:
#         T = calc_PT_profile_madhu_seager_lothringer(P,
#                                                     T3=theta[0],
#                                                     P1=10**(theta[1]),
#                                                     P3=10**(theta[2]),
#                                                     alpha2=theta[3]
#                                                    )
#         ax.plot(T, P, color=cmap.to_rgba(sigma))
#     else:
#         pass

# fig.colorbar(cmap, ticks=np.arange(0, 5, 1), label='logL normalised', extend='max')


# ax.set_yscale('log')
# ax.invert_yaxis()
# ax.set_title('example PT-profile')

# T = calc_PT_profile_madhu_seager_lothringer(P,
#                                             T3=theta_max[0],
#                                             P1=10**(theta_max[1]),
#                                             P3=10**(theta_max[2]),
#                                             alpha2=theta_max[3]
#                                            )
# #ax.plot(T, P, color='red')

# # Plot models themselves
# fname_model_list = np.array([os.path.join(dirin, f) for f in os.listdir(dirin) \
#                              if f.endswith('.csv') \
#                              and 'z1' in f \
#                              and not 'structure' in f])
# print(fname_model_list)
# #labels = [os.path.basename(f)[:-4] for f in fname_model_list]
# labels = np.array(
#     [r'$\log{z}$ = -1',
#      r'$\log{z}$ = 0',
#      r'$\log{z}$ = 1',
#      r'$\log{z}$ = -1',
#      r'$\log{z}$ = 1',
#      r'$\log{z}$ = 0']
# )
# ind_sorted = [3,1,2,0,5,4]

# if plot_phoenix:
#     cmap = cm.get_cmap('plasma_r',len(labels))
#     colors = cmap(np.linspace(0.2,0.9,len(labels)))
#     colors = [TABLEAU20[n*2] for n in range(6)]
#     for n, fname in enumerate(fname_model_list[2-2]):
#         print(fname)
#         # Load and plot PT-structure
#         structure_fname = os.path.basename(fname).split('.')
#         structure_fname = ".".join(structure_fname[:-2]) +'.structure.csv'
#         data_structure = np.genfromtxt(os.path.join(dirin, structure_fname))
#         pressure, temperature = data_structure[:,0], data_structure[:,1]
#         ax.plot(temperature, pressure, lw=2.5, alpha=1, color=colors[n], ls='-')

# plt.show()


# In[20]:


NO_TXT_EXTENSION = slice(0,-4,1)
def fit_continuum(wav, spec, nbins, method=np.min):
    spec_c, bin_edges, _ = binned_statistic(wav, spec, bins=nbins, statistic=method)
    wav_c = bin_edges[:-1] + np.diff(bin_edges)/2.
    return wav_c, spec_c


# In[32]:


for obsdate in obsdates:
    OBSERVATION_BASE_DIR = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}')
    dirout_templates = os.path.abspath(OBSERVATION_BASE_DIR + f'/processed/hrccs_templates/{dirname_models}')
    if not os.path.exists(dirout_templates):
        os.makedirs(dirout_templates)
    dirin_residuals = os.path.abspath(OBSERVATION_BASE_DIR+f'/processed/spectra/detrending/{dirname_residuals}')
    
    ntotal = len(fname_models)
    for niter, fname_model in enumerate(fname_models,1):
        print(f'Progress: {niter}/{ntotal}')
        dirout_model = os.path.join(dirout_templates, fname_model[NO_TXT_EXTENSION])
        if subtract_continuum:
            dirout_model += '_continuum_subtracted'
        if apply_hpf_to_model:
            dirout_model += '_hpf'
        if not os.path.exists(dirout_model):
            os.mkdir(dirout_model)
        print(f"Using output directory: {dirout_model}")

        # Load synthetic spectrum
        fpath_model = os.path.join(dirin_models, fname_model)
        template_wav, template_spec = load_planet_synthetic_spectrum(fpath_model, mode=load_synthetic_spectrum_mode)

        #  Create a figure of synthetic spectrum
        fig, ax = plt.subplots(figsize=(7.5*1.68, 7.5))
        plt.plot(template_wav, template_spec, lw=1, color=TABLEAU20[0])
        for temp, ls in zip([2500, 3000, 3500], ['--', '-', ':']):
            bbplanet = planck(wav=template_wav*1e-6, T=temp, unit_system='cgs')
            plt.plot(template_wav, np.pi*bbplanet, color='k', ls=ls, label=f'T={temp} K')
        plt.legend(loc=1, fontsize=15)
        plt.xlim(template_wav[0], template_wav[-1])
        plt.ylabel('Flux [erg/s/cm^2/cm]', size=15)
        plt.xlabel('Wavelength [micron]', size=15)
        plt.savefig(os.path.join(dirout_model, f'selfconsistentmodel.png'), dpi=200)
        plt.close()

        #  Add fitted continuum to plot
        fig, ax = plt.subplots(figsize=(7.5*1.68, 7.5))
        bbstar = planck(template_wav*1e-6, system['teff_star'], unit_system='cgs') # BB flux in cgs units, same units as 10**template_spec
        template_model = (template_spec/(np.pi*bbstar)) * ( (system['rp']*R_jup.value) / (system['rstar']*R_sun.value) )**2
        wav_c, spec_c = fit_continuum(template_wav, template_model, nbins=200)
        plt.plot(wav_c, spec_c, color='gray', lw=4, label='fitted continuum')

        if subtract_continuum:
            print('\tSubtracting continuum...')
            fitted_continuum = np.interp(x=template_wav, xp=wav_c, fp=spec_c)
            template_model = template_model - fitted_continuum
            print('Done.\n')

        #  Plot template spectrum used
        plt.plot(template_wav, template_model, label='template used')
        plt.xlim(template_wav[0], template_wav[-1])
        plt.ylabel('Flux [erg/s/cm^2/cm]', size=15)
        plt.xlabel('Wavelength [micron]', size=15)
        plt.title('Template model used', size =15)
        plt.savefig(os.path.join(dirout_model, 'template.png'), dpi=200)
        header = 'template wav [micron] template model flux [erg/s/cm^2/cm]'
        np.savetxt(os.path.join(dirout_model, 'template.txt'), np.c_[template_wav, template_model], header=header, delimiter=',')
        plt.close()

        # Convolve templates to measured Instrumental Profile
        templateorders = []


        print('\tConvolving template model using measured resolving power...')
        # Load IP reference order to deal with unknown measured R values for other orders
        dirin_residuals = os.path.abspath(OBSERVATION_BASE_DIR+f'/processed/spectra/detrending/{dirname_residuals}')

        dirin_ipfit = os.path.abspath(os.path.join(OBSERVATION_BASE_DIR + f'/processed/ipfit/mcmc/'))
        ipfit_values = np.loadtxt(os.path.join(dirin_ipfit, f"order_{IP_reference_order}/ipfit_result_order_{IP_reference_order}.txt"))
        measured_R_fill_values = ipfit_values[:,3]
        measured_R_values = np.zeros(len(measured_R_fill_values))
        measured_R_values[:] = np.nanmedian(measured_R_fill_values)

        # For each spectral order, convolve to the appropriate measured resolution
        for i, norder in enumerate(orders):
            # Convolve to the measured resolution
            template_npoints = len(template_model)
            nobs = len(measured_R_values)
            template_spec_convolved = np.zeros(shape=(template_npoints, nobs))
            for n, R in enumerate(measured_R_values):
                convolution_kernel = gaussian_convolution_kernel(wav=template_wav, resolution=R)
                template_spec_convolved[:, n] = np.convolve(template_model, convolution_kernel, mode='same')

            # Cut template to sufficiently large wavelength range for each order
            wavsolution = np.loadtxt(os.path.join(dirin_residuals, f'order_{norder}/{targetname}_wavsolution_{norder}.txt'))
            wavmin = wavsolution[1][0]
            wavmax = wavsolution[1][-1]
            dw = WAVELENGTH_PADDING * (wavmax-wavmin) # Add quarter x full wavelength range of padding
            wavrange_mask = np.logical_and(template_wav >= wavmin - dw, template_wav <= wavmax + dw)
            template_wav_new = template_wav[wavrange_mask]
            template_spec_convolved_new = template_spec_convolved[wavrange_mask, :]

            # Plot and save result somewhere....
            to = TemplateOrder(
                data=template_spec_convolved_new,
                wav=template_wav_new,
                norder=norder,
                R_values=measured_R_values,
                targetname=targetname,
                obsdate=obsdate
            )
            templateorders.append(to)

            #  Create overview figure for convolved template order
            fig, axes = to.plot()
            plt.savefig(os.path.join(dirout_model, f'{to.targetname}_convolved_template_order_{to.norder}.png'), dpi=200)
            plt.close()
            
            if apply_hpf_to_model:
                sampling_sf = np.median(np.diff(wavsolution[1]))/np.median(np.diff(to.wavegrid)) # difference between data and template sampling rate
                model_hpf = apply_highpass_filter(to.data.T, freq_cutoff=HIGHPASSFILTER_FREQ_CUTOFF / sampling_sf).T
                to.data = model_hpf
                
                fig, axes = to.plot()
                plt.savefig(os.path.join(dirout_model, f'{to.targetname}_convolved_hpf_template_order_{to.norder}.png'), dpi=200)
                plt.close()
            
            to.save(os.path.join(dirout_model, f'{to.targetname}_template_order_{to.norder}'), data_only=False)
        print("Done.\n")


# In[6]:


# for obsdate in obsdates:
#     OBSERVATION_BASE_DIR = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}')
#     dirout_templates = os.path.abspath(OBSERVATION_BASE_DIR + f'/processed/hrccs_templates/{dirname_models}')
#     if not os.path.exists(dirout_templates):
#         os.makedirs(dirout_templates)
    
#     for fname_model in fname_models:
#         dirout_model = os.path.join(dirout_templates, fname_model[NO_TXT_EXTENSION])
#         if subtract_continuum:
#             dirout_model += '_continuum_subtracted'
#         if not os.path.exists(dirout_model):
#             os.mkdir(dirout_model)
#         print(f"Using output directory: {dirout_model}")
        
#         # Load synthetic spectrum
#         fpath_model = os.path.join(dirin_models, fname_model)
#         template_wav, template_spec = load_planet_synthetic_spectrum(fpath_model, mode=load_synthetic_spectrum_mode)
        
#         #  Create a figure of synthetic spectrum
#         fig, ax = plt.subplots(figsize=(7.5*1.68, 7.5))
#         plt.plot(template_wav, template_spec, lw=1, color=TABLEAU20[0])
#         for temp, ls in zip([2500, 3000, 3500], ['--', '-', ':']):
#             bbplanet = planck(wav=template_wav*1e-6, T=temp, unit_system='cgs')
#             plt.plot(template_wav, np.pi*bbplanet, color='k', ls=ls, label=f'T={temp} K')
#         plt.legend(loc=1, fontsize=15)
#         plt.xlim(template_wav[0], template_wav[-1])
#         plt.ylabel('Flux [erg/s/cm^2/cm]', size=15)
#         plt.xlabel('Wavelength [micron]', size=15)
#         plt.savefig(os.path.join(dirout_model, f'selfconsistentmodel.png'), dpi=200)
#         plt.close()
        
#         #  Add fitted continuum to plot
#         fig, ax = plt.subplots(figsize=(7.5*1.68, 7.5))
#         bbstar = planck(template_wav*1e-6, system['teff_star'], unit_system='cgs') # BB flux in cgs units, same units as 10**template_spec
#         template_model = (template_spec/(np.pi*bbstar)) * ( (system['rp']*R_jup.value) / (system['rstar']*R_sun.value) )**2
#         wav_c, spec_c = fit_continuum(template_wav, template_model, nbins=200)
#         plt.plot(wav_c, spec_c, color='gray', lw=4, label='fitted continuum')
        
#         if subtract_continuum:
#             print('\tSubtracting continuum...')
#             fitted_continuum = np.interp(x=template_wav, xp=wav_c, fp=spec_c)
#             template_model = template_model - fitted_continuum
#             print('Done.\n')

#         #  Plot template spectrum used
#         plt.plot(template_wav, template_model, label='template used')
#         plt.xlim(template_wav[0], template_wav[-1])
#         plt.ylabel('Flux [erg/s/cm^2/cm]', size=15)
#         plt.xlabel('Wavelength [micron]', size=15)
#         plt.title('Template model used', size =15)
#         plt.savefig(os.path.join(dirout_model, 'template.png'), dpi=200)
#         header = 'template wav [micron] template model flux [erg/s/cm^2/cm]'
#         np.savetxt(os.path.join(dirout_model, 'template.txt'), np.c_[template_wav, template_model], header=header, delimiter=',')
#         plt.close()
        
#         # Convolve templates to measured Instrumental Profile
#         templateorders = []


#         print('\tConvolving template model using measured resolving power...')
#         # Load IP reference order to deal with unknown measured R values for other orders
#         dirin_residuals = os.path.abspath(OBSERVATION_BASE_DIR+f'/processed/spectra/detrending/{dirname_residuals}')
        
#         dirin_ipfit = os.path.abspath(os.path.join(OBSERVATION_BASE_DIR + f'/processed/ipfit/mcmc/'))
#         ipfit_values = np.loadtxt(os.path.join(dirin_ipfit, f"order_{IP_reference_order}/ipfit_result_order_{IP_reference_order}.txt"))
#         measured_R_fill_values = ipfit_values[:,3]
#         measured_R_values = np.zeros(len(measured_R_fill_values))
#         measured_R_values[:] = np.nanmedian(measured_R_fill_values)

#         # For each spectral order, convolve to the appropriate measured resolution
#         for i, norder in enumerate(orders):
#             # Convolve to the measured resolution
#             template_npoints = len(template_model)
#             nobs = len(measured_R_values)
#             template_spec_convolved = np.zeros(shape=(template_npoints, nobs))
#             for n, R in enumerate(measured_R_values):
#                 convolution_kernel = gaussian_convolution_kernel(wav=template_wav, resolution=R)
#                 template_spec_convolved[:, n] = np.convolve(template_model, convolution_kernel, mode='same')

#             # Cut template to sufficiently large wavelength range for each order
#             wavsolution = np.loadtxt(os.path.join(dirin_residuals, f'order_{norder}/{targetname}_wavsolution_{norder}.txt'))
#             wavmin = wavsolution[1][0]
#             wavmax = wavsolution[1][-1]
#             dw = WAVELENGTH_PADDING * (wavmax-wavmin) # Add quarter x full wavelength range of padding
#             wavrange_mask = np.logical_and(template_wav >= wavmin - dw, template_wav <= wavmax + dw)
#             template_wav_new = template_wav[wavrange_mask]
#             template_spec_convolved_new = template_spec_convolved[wavrange_mask, :]

#             # Plot and save result somewhere....
#             to = TemplateOrder(
#                 data=template_spec_convolved_new,
#                 wav=template_wav_new,
#                 norder=norder,
#                 R_values=measured_R_values,
#                 targetname=targetname,
#                 obsdate=obsdate
#             )
#             templateorders.append(to)
            
#             #  Create overview figure for convolved template order
#             fig, axes = to.plot()
#             plt.savefig(os.path.join(dirout_model, f'{to.targetname}_convolved_template_order_{to.norder}.png'), dpi=200)
#             plt.close()

#             to.save(os.path.join(dirout_model, f'{to.targetname}_template_order_{to.norder}'), data_only=False)
#         print("Done.\n")


# In[7]:


# for obsdate in obsdates:
#     OBSERVATION_BASE_DIR = os.path.abspath(ARIES_BASE_DIR + f'/data/{targetname}/{obsdate}')
#     dirout_templates = os.path.abspath(OBSERVATION_BASE_DIR + f'/processed/hrccs_templates/{dirname_models}')
#     if not os.path.exists(dirout_templates):
#         os.makedirs(dirout_templates)
    
#     for fname_model in fname_models:
#         dirout_model = os.path.join(dirout_templates, fname_model[NO_TXT_EXTENSION])
#         if subtract_continuum:
#             dirout_model += '_continuum_subtracted'
#         if not os.path.exists(dirout_model):
#             os.mkdir(dirout_model)
#         print(f"Using output directory: {dirout_model}")
        
#         # Load synthetic spectrum
#         fpath_model = os.path.join(dirin_models, fname_model)
#         template_wav, template_spec = load_planet_synthetic_spectrum(fpath_model, mode=load_synthetic_spectrum_mode)
        
#         #  Create a figure of synthetic spectrum
#         fig, ax = plt.subplots(figsize=(7.5*1.68, 7.5))
#         plt.plot(template_wav, template_spec, lw=1, color=TABLEAU20[0])
#         for temp, ls in zip([2500, 3000, 3500], ['--', '-', ':']):
#             bbplanet = planck(wav=template_wav*1e-6, T=temp, unit_system='cgs')
#             plt.plot(template_wav, np.pi*bbplanet, color='k', ls=ls, label=f'T={temp} K')
#         plt.legend(loc=1, fontsize=15)
#         plt.xlim(template_wav[0], template_wav[-1])
#         plt.ylabel('Flux [erg/s/cm^2/cm]', size=15)
#         plt.xlabel('Wavelength [micron]', size=15)
#         plt.savefig(os.path.join(dirout_model, f'selfconsistentmodel.png'), dpi=200)
#         plt.close()
        
#         #  Add fitted continuum to plot
#         fig, ax = plt.subplots(figsize=(7.5*1.68, 7.5))
#         bbstar = planck(template_wav*1e-6, system['teff_star'], unit_system='cgs') # BB flux in cgs units, same units as 10**template_spec
#         template_model = (template_spec/(np.pi*bbstar)) * ( (system['rp']*R_jup.value) / (system['rstar']*R_sun.value) )**2
#         wav_c, spec_c = fit_continuum(template_wav, template_model, nbins=200)
#         plt.plot(wav_c, spec_c, color='gray', lw=4, label='fitted continuum')
        
#         if subtract_continuum:
#             print('\tSubtracting continuum...')
#             fitted_continuum = np.interp(x=template_wav, xp=wav_c, fp=spec_c)
#             template_model = template_model - fitted_continuum
#             print('Done.\n')

#         #  Plot template spectrum used
#         plt.plot(template_wav, template_model, label='template used')
#         plt.xlim(template_wav[0], template_wav[-1])
#         plt.ylabel('Flux [erg/s/cm^2/cm]', size=15)
#         plt.xlabel('Wavelength [micron]', size=15)
#         plt.title('Template model used', size =15)
#         plt.savefig(os.path.join(dirout_model, 'template.png'), dpi=200)
#         header = 'template wav [micron] template model flux [erg/s/cm^2/cm]'
#         np.savetxt(os.path.join(dirout_model, 'template.txt'), np.c_[template_wav, template_model], header=header, delimiter=',')
#         plt.close()
        
#         # Convolve templates to measured Instrumental Profile
#         templateorders = []


#         print('\tConvolving template model using measured resolving power...')
#         # Load IP reference order to deal with unknown measured R values for other orders
#         dirin_residuals = os.path.abspath(OBSERVATION_BASE_DIR+f'/processed/spectra/detrending/{dirname_residuals}')
        
#         dirin_ipfit = os.path.abspath(os.path.join(OBSERVATION_BASE_DIR + f'/processed/ipfit/mcmc/'))
#         ipfit_values = np.loadtxt(os.path.join(dirin_ipfit, f"order_{IP_reference_order}/ipfit_result_order_{IP_reference_order}.txt"))
#         measured_R_fill_values = ipfit_values[:,3]
#         measured_R_values = np.zeros(len(measured_R_fill_values))
#         measured_R_values[:] = np.nanmedian(measured_R_fill_values)

#         # For each spectral order, convolve to the appropriate measured resolution
#         for i, norder in enumerate(orders):
#             # Convolve to the measured resolution
#             template_npoints = len(template_model)
#             nobs = len(measured_R_values)
#             template_spec_convolved = np.zeros(shape=(template_npoints, nobs))
#             for n, R in enumerate(measured_R_values):
#                 convolution_kernel = gaussian_convolution_kernel(wav=template_wav, resolution=R)
#                 template_spec_convolved[:, n] = np.convolve(template_model, convolution_kernel, mode='same')

#             # Cut template to sufficiently large wavelength range for each order
#             wavsolution = np.loadtxt(os.path.join(dirin_residuals, f'order_{norder}/{targetname}_wavsolution_{norder}.txt'))
#             wavmin = wavsolution[1][0]
#             wavmax = wavsolution[1][-1]
#             dw = WAVELENGTH_PADDING * (wavmax-wavmin) # Add quarter x full wavelength range of padding
#             wavrange_mask = np.logical_and(template_wav >= wavmin - dw, template_wav <= wavmax + dw)
#             template_wav_new = template_wav[wavrange_mask]
#             template_spec_convolved_new = template_spec_convolved[wavrange_mask, :]

#             # Plot and save result somewhere....
#             to = TemplateOrder(
#                 data=template_spec_convolved_new,
#                 wav=template_wav_new,
#                 norder=norder,
#                 R_values=measured_R_values,
#                 targetname=targetname,
#                 obsdate=obsdate
#             )
#             templateorders.append(to)
            
#             #  Create overview figure for convolved template order
#             fig, axes = to.plot()
#             plt.savefig(os.path.join(dirout_model, f'{to.targetname}_convolved_template_order_{to.norder}.png'), dpi=200)
#             plt.close()

#             to.save(os.path.join(dirout_model, f'{to.targetname}_template_order_{to.norder}'), data_only=False)
#         print("Done.\n")


# ## Loading template orders
# 
# Now check if we can correctly load these template orders.

# In[13]:


# from imp import reload
# import aries
# reload(aries.cleanspec)
# from aries.cleanspec import TemplateOrder

# dirin_models = os.path.abspath(ARIES_BASE_DIR + "models/wasp33/w33_gcm_elsie")
# fpath_model = os.path.join(dirin_models, "Em_0.0_template_all.txt")
# dirin_templates = os.path.abspath(OBSERVATION_BASE_DIR + \
#                                   '/processed/hrccs_templates/w33_gcm_elsie/' + \
#                                   os.path.basename(fpath_model)[NO_TXT_EXTENSION])

# templateorders = [TemplateOrder.load(f=os.path.join(dirin_templates, f'{targetname}_template_order_{norder}')) for norder in orders]


# In[12]:


# for to in templateorders:
#     to.plot()
#     plt.show()


# In[ ]:




