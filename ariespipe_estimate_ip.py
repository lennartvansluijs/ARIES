#!/usr/bin/env python
# coding: utf-8

# # Post-processing: estimating the Instrumental Profile (IP)
# ---

# <b>Modules and packages

# In[1]:

import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings

from astropy.io import fits
from astropy.time import Time
import sys
from settings import ARIES_BASE_DIR, DATA_BASE_DIR
sys.path.append(ARIES_BASE_DIR)

from aries.cleanspec import SpectralOrder, clip_mask
from aries.constants import TABLEAU20, ARIES_NORDERS
from aries.crosscorrelate import get_rvplanet
from aries.preprocessing import get_keyword_from_headers_in_dir, get_fits_fnames


# <b>Target info

# In[2]:


#from aries.systemparams import systems, targetname_to_key

badorders = []
orders = [n for n in range(1, ARIES_NORDERS+1) if n not in badorders]
orders = [2] # use this order as a test case for now
#orders = [23]

# ---
# Parse input parameters
# ---


parser = argparse.ArgumentParser()
parser.add_argument('-targetname')
parser.add_argument('-obsdate')

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate
#print(targetname, obsdate)
# targetname = 'wasp33'
#targetid = targetname_to_key(targetname)
#systemparams = systems[targetid]

# obsdate = '20161015'


OBSERVATION_BASE_DIR = os.path.abspath(DATA_BASE_DIR + '/{}/{}'.format(targetname, obsdate))


# <b>Input directories<b>

# In[3]:


#dirin_wavcal = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/wavcal')
dirin_wavcal = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/autowavcal')
dirin_data = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/spectra/alignment_with_stretching')

# <b>Algorithm parameters

# In[4]:


do_build_telfit_lib = False#True
do_mcmc_ipfit = True# True


# ## Load planet radial velocity data and time stamps

# In[5]:


# dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/meta')
# fpath = os.path.join(dirin, '{}_targetinfo.csv'.format(targetname))
# with open(fpath, mode='r') as infile:
#     reader = csv.reader(infile)
#     systemparams = {rows[0]:float(rows[1]) for rows in reader}




# ## Load aligned spectral orders

# In[6]:


data_all = [fits.getdata(os.path.join(dirin_data, 'order_{}/{}_order_aligned.fits'.format(norder,targetname))) for norder in orders]
mask_all = [fits.getdata(os.path.join(dirin_data, 'order_{}/{}_order_mask.fits'.format(norder, targetname))) for norder in orders]
wavsolution_all = [np.loadtxt(os.path.join(dirin_wavcal, '{}_wavcal_order_{}_wavsolution.txt'.format(targetname, norder))).T for norder in orders]

spectralorders = []
for (norder, data, mask, wavsolution) in zip(orders, data_all, mask_all, wavsolution_all):

    data_c = clip_mask(data, mask)
    mask_c = np.zeros(data_c.shape)

    so = SpectralOrder(data=data_c, mask=mask_c, norder=norder,
    obsdate=obsdate, wavsolution=wavsolution, target=targetname,
    phase=None, time=None)
    spectralorders.append(so)


# Plot all aligned and wavelength calibrated spectral orders

# In[7]:


for so in spectralorders:
    so.plot(xunit='micron')
    plt.show()


# In[8]:


#from aries.ipfit import correct_continuum

# wav, spec = so.wavsolution[1], so.data[0,:]
# spec_corr = correct_continuum(wav, spec, do_plot=False, polydeg=3) # correct for slope in the continuum

# dirout = "/home/lennart/measure/models/telluric/resolution_check"
# np.save(os.path.join(dirout, 'wav_example.npy'), wav)
# np.save(os.path.join(dirout, 'spec_example.npy'), spec)
# np.save(os.path.join(dirout, 'spec_corr_example.npy'), spec_corr)
# plt.plot(wav, spec_corr)
# plt.show()


# ## Build a Telluric Library

# First let's specify the values for which we want Telfit to build telluric models.

# In[9]:


from aries.ipfit import correct_continuum
from aries.ipfit import arange_at_fixed_R
from aries.ipfit import convert_airmass_to_angle

data_all = [fits.getdata(os.path.join(dirin_data, 'order_{}/{}_order_aligned.fits'.format(norder,targetname))) for norder in orders]
mask_all = [fits.getdata(os.path.join(dirin_data, 'order_{}/{}_order_mask.fits'.format(norder, targetname))) for norder in orders]
wavsolution_all = [np.loadtxt(os.path.join(dirin_wavcal, '{}_wavcal_order_{}_wavsolution.txt'.format(targetname, norder))).T for norder in orders]

spectralorders = []
for (norder, data, mask, wavsolution) in zip(orders, data_all, mask_all, wavsolution_all):
    data_c = clip_mask(data, mask)
    mask_c = np.zeros(data_c.shape)
    so = SpectralOrder(data=data_c, mask=mask_c, norder=norder,
    wavsolution=wavsolution, target=targetname, phase=None, time=None)
    so.obsdate = obsdate
    spectralorders.append(so)

# Load airmass data
try:
    fpath_airmass = os.path.join(OBSERVATION_BASE_DIR+'/meta/airmass.txt')
    airmass_values = np.loadtxt(fpath_airmass)
    print('bleh')
except:
    DAY = 24*60*60 # s
    dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/raw')
    fnames = np.array(get_fits_fnames(dirin, key='science'))
    exptimes = get_keyword_from_headers_in_dir(keyword='EXPTIME', dirname=dirin, key='science')
    times_utstart = get_keyword_from_headers_in_dir(keyword='UTSTART', dirname=dirin, key='science')
    airmass_values = get_keyword_from_headers_in_dir(keyword='AIRMASS', dirname=dirin, key='science')

    # conversion of times to JD and BJD
    times_jdutc = np.array([Time(t, scale='utc').jd for t in times_utstart]) + (exptimes/2.) / DAY
    TIMES_SORTED = np.argsort(times_jdutc)

    # also save airmass for later usage
    dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/meta')
    fpathout = os.path.join(dirout,'airmass.txt')
    np.savetxt(fpathout, airmass_values[TIMES_SORTED])

for so in spectralorders:
    if len(airmass_values) != so.nobs:
        raise ValueError('Airmass values must be equal to number of observations.')
    so.airmass_values = airmass_values
    so.ZA_values = convert_airmass_to_angle(airmass_values)


# Optional: not sure if this helps, but I will try to correct continuum slope.

# Next let's define a wavelenght grid with a stepsize at a fixed spectral resolution.

# Finally, we can build our library for all these spectral orders.

# In[10]:


from aries.ipfit import build_telluric_lib

observatory_params = {
    'lat' : 31, # degree
    'alt' : 2.16, #km
    'Tsurface' : 298 # let's assume a fixed surface temperature of 298 Kelvin for now,
    # may want to change this later on, was this recorded somewhere?
}

zenithangle_values =  np.arange(0,90,15)
humidity_values = np.array([0.5,1.,2.,5.,10.,25,50,100.])
dw = 100 # add nm of padding on each side
npoints = int(1e4) # needs to be higher than data sampling

if do_build_telfit_lib:
    for so in spectralorders:
        print(f'Building Telfit library for order={so.norder}')
        wavmin = so.wavsolution[1][0]*1e3 - dw # nm
        wavmax = so.wavsolution[1][-1]*1e3 + dw # nm
        print(wavmin, wavmax, npoints)


        # it should at least be higher sampling than real data
        wavgrid_os = np.linspace(wavmin, wavmax, npoints)

        dirout = os.path.abspath(ARIES_BASE_DIR+f'/models/telfit/{targetname}/order{so.norder}')
        if not os.path.exists(dirout):
            os.makedirs(dirout)
        build_telluric_lib(zenithangle_values, humidity_values, observatory_params, wavgrid_os, dirout)


# ## IP fitting using MCMC

# In[ ]:


from aries.ipfit import convert_RH_to_PWV
from aries.ipfit import locate_2d_max
from aries.ipfit import TelluricInterpolator
import scipy.optimize as opt
from aries.crosscorrelate import calc_ccmatrix
from multiprocessing import Pool, cpu_count
import emcee
import pickle
from scipy.optimize import minimize
os.environ["OMP_NUM_THREADS"] = "1"


class InstrumentalProfileMCMCSampler:
    """"""
    def __init__(self, wav_data, spec_data, ZA_data, RH_range=(0,100), R_range=(1e4, 3.5e4)):
        """"""
        self.RH_range = RH_range # Relative Humidity bounds
        self.R_range = R_range # Spectral Resolution bounds
        self.ZA_data = ZA_data # Zenith Angle at which spectrum is observed
        self.wav_data, self.spec_data = wav_data, spec_data # Observed spectrum and wavelength range

    def log_prior(self, theta):
        """"""
        RH, R = theta
        if self.RH_range[0] < RH < self.RH_range[1] and self.R_range[0] < R < self.R_range[1]:
            print('yes')
            return 0.
        else:
            print('no')
            return -np.inf

    def log_probability(self, theta):
        """"""
        return self.logL_zucker(theta)

    def logL_zucker(self, theta):
        """"""
        RH, R = theta
        _, telluric_model = interp_telluric(
                self.wav_data*1e3, # convert to nm
                zenithangle=self.ZA_data,
                humidity=RH,
                R=R
        )
        cc = np.corrcoef(self.spec_data, telluric_model)[0,1]
        N = len(self.spec_data)
        logL = -0.5*N*np.log(1-cc**2)
        if not np.isnan(logL):
            return -np.inf

    def run(self, theta_prior, nwalkers=8, nsteps=1e2, use_pool=False):
        """"""
        ndim = len(theta_prior)
        pos = theta_prior + 1e-4 * np.random.randn(nwalkers, ndim)
        if use_pool:
            with Pool() as p:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, pool=p)
                sampler.run_mcmc(pos, nsteps, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
            sampler.run_mcmc(pos, nsteps, progress=True)
        return sampler

class InstrumentalProfileMCMCSampler:
    """"""
    def __init__(self, wav_data, spec_data, ZA_data, RH_range=(0,100), R_range=(1e4, 3.5e4)):
        """"""
        self.RH_range = RH_range # Relative Humidity bounds
        self.R_range = R_range # Spectral Resolution bounds
        self.ZA_data = ZA_data # Zenith Angle at which spectrum is observed
        self.wav_data, self.spec_data = wav_data, spec_data # Observed spectrum and wavelength range

    def log_prior(self, theta):
        """"""
        RH, R = theta
        if self.RH_range[0] < RH < self.RH_range[1] and self.R_range[0] < R < self.R_range[1]:
            return 0.
        else:
            return -np.inf

    def log_probability(self, theta):
        """"""
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + logL_zucker(theta)

    def logL_zucker(self, theta):
        """"""
        RH, R = theta
        _, telluric_model = interp_telluric(
                self.wav_data*1e3, # convert to nm
                zenithangle=self.ZA_data,
                humidity=RH,
                R=R
        )
        BADVALUES = np.isnan(telluric_model)
        telluric_model[BADVALUES] = np.nanmedian(telluric_model)
        cc = np.corrcoef(self.spec_data, telluric_model)[0,1]
        N = len(self.spec_data)
        #lp = self.log_prior(theta)
        logL = -0.5*N*np.log(1-cc**2)
        if (~np.isnan(logL) or ~np.isfinite(logL)):
            return logL
        else:
            return 0.

    def run(self, theta_prior, nwalkers=8, nsteps=1e2, use_pool=False):
        """"""
        ndim = len(theta_prior)
        pos = theta_prior + 1e-4 * np.random.randn(nwalkers, ndim)
        if use_pool:
            with Pool() as p:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logL_zucker, pool=p)
                sampler.run_mcmc(pos, nsteps, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logL_zucker)
            sampler.run_mcmc(pos, nsteps, progress=True)
        return sampler

def plot_walkers(samples, ndim=2):
    """"""
    ndim = samples.shape[-1]
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

    labels = ["RH", "Resolution / 10,000"]
    sf_values = [1, 1e4]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i]/sf_values[i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i], size=15)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step number", size=15)

def plot_corner(flat_samples):
    labels = ["RH", "Resolution"]
    import corner
    fig = corner.corner(
        flat_samples, labels=labels
    )

def get_result_mcmc(flat_samples, verbose=True):
    from IPython.display import display, Math

    params = {}
    labels = ['RH', 'R']
    for i, key in enumerate(labels):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)

        if verbose:
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            display(Math(txt))

        params[key] = (mcmc[1], q[0], q[1])
    return params


def do_ipfit_all_mcmc(spectralorders, mcmc_params, dirout, do_continuumcorr=False, **kwargs):
    """"""
    for so in spectralorders:
        dirout_order = os.path.abspath(os.path.join(dirout, f'order_{so.norder}'))
        if not os.path.exists(dirout_order):
            os.makedirs(dirout_order)
        print(f'IP fit for spectral order {so.norder}')

        # This is prob not the most stable solution, but it works for now
        global interp_telluric
        interp_telluric = TelluricInterpolator(dirin=os.path.abspath(ARIES_BASE_DIR+f'/models/telfit/{targetname}/order{so.norder}'))

        ipfit_values = np.zeros(shape=(so.nobs, 6)) + np.nan
        wav = so.wavsolution[1]
        for n in range(so.nobs):
            ZA = so.ZA_values[n] # Load Zenith Angle for this observation

            dirout_frame = os.path.join(dirout_order, f'frame_{n+1}')
            print(f'IP fit frame: {n+1}/{so.nobs}')
            if not os.path.exists(dirout_frame):
                os.mkdir(dirout_frame)
            if do_continuumcorr:
                # Try to correct for the continuum by fitting a polynomial to maxima in each bin
                spec = correct_continuum(wav, so.data[n,:], do_plot=True, **kwargs)
                plt.savefig(os.path.join(dirout_frame, 'ipfit_continuumcorr.png'), dpi=200)
                plt.close()
            else:
                spec = so.data[n,:]

            try:
                ipsampler = InstrumentalProfileMCMCSampler(wav, spec, ZA)
                result = ipsampler.run(theta_prior=mcmc_params['theta_prior'],
                                       nwalkers=mcmc_params['nwalkers'],
                                       nsteps=mcmc_params['nsteps'],
                                       use_pool=mcmc_params['use_pool'])

                samples = result.get_chain()
                plot_walkers(samples)
                plt.savefig(os.path.join(dirout_frame, 'mcmc_walkers.png'), dpi=200)
                np.save(os.path.join(dirout_frame, 'samples.npy'), samples)
                plt.close()

                flat_samples = result.get_chain(discard=mcmc_params['burn_in'], flat=True)
                plot_corner(flat_samples)
                plt.savefig(os.path.join(dirout_frame, 'mcmc_corner.png'), dpi=200)
                np.save(os.path.join(dirout_frame, 'flat_samples.npy'), flat_samples)
                plt.close()

                params_result = get_result_mcmc(flat_samples)
                f = os.path.join(dirout_frame, 'ipfit_result.pickle')
                with open(f, 'wb') as handle:
                    pickle.dump(params_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # Save best values
                ipfit_values[n, :3] = params_result["RH"]
                ipfit_values[n, 3:] = params_result['R']

                # Finally plot the output of the best telluric fit
                _, telluric_bestmodel = interp_telluric(
                        wav*1e3, # convert to nm
                        zenithangle=ZA,
                        humidity=params_result["RH"][0],
                        R=params_result["R"][0]
                )
                _, telluric_bestmodel_prior = interp_telluric(
                        wav*1e3, # convert to nm
                        zenithangle=ZA,
                        humidity=50.,
                        R=2e4
                )
                plt.plot(wav, telluric_bestmodel, color=TABLEAU20[2], label='bestmodel')
                plt.plot(wav, spec, color=TABLEAU20[0], label='observed spectrum')
                plt.plot(wav, telluric_bestmodel_prior, color=TABLEAU20[4], label='prior')
                plt.legend()
                plt.savefig(os.path.join(dirout_frame, 'telluric_bestmodel.png'), dpi=200)
                plt.close()

            except:
                warnings.warn(f'\nFailed to find a solution for frame = {n+1}. '
                              'Setting measured values equal to NaN.')
            finally:
                print('Done.\n')
                # Save all
                fpathout = os.path.join(dirout_order, f'ipfit_result_order_{so.norder}.txt')
                hdr = 'R, RH'
                np.savetxt(fpathout, ipfit_values, header=hdr)

# Define initial guess
RH_prior = 50.
R_prior = 2e4
theta_prior = (RH_prior, R_prior)

mcmc_params = {
    "theta_prior" : theta_prior,
    "nwalkers" : 8,
    "nsteps" : int(1e3),
    "burn_in" : int(1e2),
    "use_pool" : True,
}

spectralorders_list = [so for i, so in enumerate(spectralorders) if not so.norder in np.arange(23)]
# print(spectralorders[0].data.shape)
# print(spectralorders[0].phase.size)
# print(spectralorders[0].wavsolution[0].size)
dirout_ipfit = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/ipfit/mcmc/')
if not os.path.exists(dirout_ipfit):
    os.makedirs(dirout_ipfit)
if do_mcmc_ipfit:
    do_ipfit_all_mcmc(spectralorders, mcmc_params, dirout=dirout_ipfit, do_continuumcorr=True, polydeg=3)


# Reload all results from individual frames.

# ## Calculate throughput values

# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from itertools import product
so = spectralorders[0]

# calculate relative throughput and errorbars
throughput_values = np.mean(so.data/np.median(so.data, axis=0), axis=1) # mean flux in median normalised spectrum
throughput_err_values = np.std(so.data/np.median(so.data, axis=0), axis=1) # std flux in median normalised spectrum

fpathout = os.path.abspath(os.path.join(dirout_ipfit, f'order_{so.norder}/throughput'))
np.savetxt(fpathout+'.txt', np.array([throughput_values, throughput_err_values]).T)

frames = np.arange(1, so.nobs+1)
plt.errorbar(frames, throughput_values, throughput_err_values)
plt.ylabel('Relative throughput')
plt.xlabel('# Frame')
plt.savefig(fpathout+'.png', dpi=200)


# ## Plot measured spectral resolution + PWV

# In[ ]:


def plot_ipfit_result(ipfit_values):
    """PLot MCMC result of fitting the IP profile to the spectra of one order."""
    nframes = ipfit_values.shape[0]

    PWV_values = np.array([convert_RH_to_PWV(RH, Tsurf=observatory_params['Tsurface']) for RH in ipfit_values[:,0]])
    PWV_lb_values = np.array([convert_RH_to_PWV(RH-RH_err, Tsurf=observatory_params['Tsurface']) for RH, RH_err in zip(ipfit_values[:,0], ipfit_values[:,1])])
    PWV_ub_values = np.array([convert_RH_to_PWV(RH-RH_err, Tsurf=observatory_params['Tsurface']) for RH, RH_err in zip(ipfit_values[:,0], ipfit_values[:,2])])
    PWV_err = (np.abs(PWV_values-PWV_lb_values), np.abs(PWV_values-PWV_ub_values))

    fig, axes = plt.subplots(2, figsize=(7.5, 7.5), sharex=True)
    nframes = ipfit_values.shape[0]
    axes[0].errorbar(x=np.arange(nframes), y=PWV_values, yerr=PWV_err, color=TABLEAU20[0])
    axes[0].set_ylabel('PWV [mm]', size=15)
    axes[0].set_ylim(np.floor(np.nanmin(PWV_values-PWV_err)), np.ceil(np.nanmax(PWV_values+PWV_err)))
    axes[0].set_xlim(-1, nframes)

    axes[1].errorbar(x=np.arange(nframes), y=ipfit_values[:,3], yerr=(ipfit_values[:,4], ipfit_values[:,5]), color=TABLEAU20[2])
    axes[1].set_ylabel('Relative spectral resolution', size=15)
    axes[1].set_ylim(1.e4, 2.5e4)
    axes[1].set_xlabel('Frame number', size=15)
    axes[1].set_xlim(-1, nframes)
    return fig, axes

for obsdate in [obsdate]:
    OBSERVATION_BASE_DIR = os.path.abspath(DATA_BASE_DIR + '/{}/{}'.format(targetname, obsdate))
    for norder in orders:
        dirin = os.path.abspath(os.path.join(OBSERVATION_BASE_DIR + f'/processed/ipfit/mcmc/order_{norder}'))
        ipfit_values = np.loadtxt(os.path.join(dirin, f"ipfit_result_order_{norder}.txt"))
        print(ipfit_values, ipfit_values.shape)
        fig, axes = plot_ipfit_result(ipfit_values)
        axes[0].set_title(f'norder={norder}, night={obsdate}', size=15)
        plt.savefig(os.path.join(dirin, 'ipfit_measured_pwv_resolution.png'), dpi=200)
        #plt.show()
        print('Resolution:', ipfit_values[:,3])


# In[ ]:
