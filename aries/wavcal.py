from scipy.interpolate import interp1d
from .cleanspec import gaussian
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from .preprocessing import robust_polyfit
from .constants import TABLEAU20

from .ipfit import convert_RH_to_PWV
from .ipfit import locate_2d_max
from .ipfit import TelluricInterpolator
from .crosscorrelate import calc_ccmatrix
from multiprocessing import Pool, cpu_count
import emcee
import pickle
import corner
from scipy.optimize import minimize

from scipy import polyval
from .cleanspec import clip_mask
from .ipfit import correct_continuum
from itertools import product

NCLOSEST_DATA = 7
NCLOSEST_MODEL = 25

def get_closest_min(x, y, value):
    ind_closest = np.argsort(np.abs(x-value))[0]
    ind_min = argrelextrema(y, np.less)[0]
    ind_closest_min = ind_min[np.argsort(np.abs(ind_min-ind_closest))][0]
    return x[ind_closest_min], y[ind_closest_min]

def get_n_closest_neighbours(x, y, value, n):
    ind = np.argsort(np.abs(x-value))
    n_closest = np.sort(ind[:n])
    return x[n_closest], y[n_closest]

def gaussian(x, *p):
    amp, mu, sigma, y0 = p
    return y0+amp*np.exp(-(x-mu)**2/(2.*sigma**2))

def find_location_spectral_lines(x_estimate, y_estimate, wav, spec, mode='cubic', ax=None, osf=100, plot=True, nclosest=5):
    xfit, yfit = [], []
    for xp, yp in zip(x_estimate, y_estimate):
        
        xmin, ymin = get_closest_min(wav, spec, value=xp)
        neighbours = get_n_closest_neighbours(wav, spec, value=xmin, n=nclosest)

        if mode == 'gaussian':
            try:
                mu0 = neighbours[0][np.argmin(neighbours[1])]
                amp0 = neighbours[1].min() - 1.
                sigma0 = (neighbours[0].max() - neighbours[0].min())
                y0 = 1.
                p0 = [amp0, mu0, sigma0, y0]
                coefs, var_matrix = curve_fit(gaussian, neighbours[0], neighbours[1], p0)
                xos = plt.linspace(neighbours[0][0], neighbours[0][-1], osf)
                yos = gaussian(xos, *coefs)
            except:
                # gaussian fit undefined, falling back on cubic interpolation
                mode = 'cubic'
        
        if mode == 'cubic':
            f = interp1d(x=neighbours[0], y=neighbours[1], kind='cubic')
            xos = plt.linspace(neighbours[0][0], neighbours[0][-1], osf)
            yos = f(xos)
        
        INDEX_MIN = np.argmin(yos)
        xfit.append(xos[INDEX_MIN])
        yfit.append(yos[INDEX_MIN])

        if plot:
            ax.plot(xos[INDEX_MIN], yos[INDEX_MIN], 'o',color='r',ms=5, picker=5,label='fit_min')
            ax.plot(xos, yos, color='r',label='fit', picker=5)
            plt.draw()
    
    if plot:
        for artist in ax.get_children():
            if hasattr(artist,'get_label') and artist.get_label()=='data_pnt':
                artist.remove()
                plt.draw()
                
import pylab as plt
import numpy as np
from scipy.interpolate import splrep,splev
from scipy import polyval
import sys
import os
from matplotlib import gridspec
import time

def plot_wavcal_polyfit(x_data, x_model, data_wav, norder, polydeg=3, ax=None):
    """"""
    if ax is None:
        fig, ax = plt.subplots()

    _, coefs, _ = robust_polyfit(x=x_data,
                                 y=x_model,
                                 deg=polydeg,
                                 return_full=True)
    data_wavsolution = polyval(coefs, data_wav)
    pnt_wavsolution = polyval(coefs, x_data)

    ax.set_title('Wavelength solution (order={}, deg={})'.format(norder, polydeg), size=15)
    ax.plot(x_data, x_model, 'x', ms=10, color='k')
    ax.set_xlabel('X [pixel]', size=15)
    ax.set_ylabel(r'Wavelength [micron]', size=15)
    ax.plot(data_wav, data_wavsolution, color=TABLEAU20[6], zorder=0)

    return data_wavsolution, pnt_wavsolution, coefs, ax
    
class WavCalGUI:
    
    def __init__(self, data_wav, data_spec, telluric_wav, telluric_spec, norder, fpathout, **kwargs):
        """"""
        self.data_wav, self.data_spec = data_wav, data_spec
        self.telluric_wav, self.telluric_spec = telluric_wav, telluric_spec
        self.norder = norder
        self.fpathout = fpathout
        self.polydeg = kwargs.pop('polydeg', 3)
        
    def tellme(self, s):
        plt.suptitle(s, fontsize=16)
        plt.draw()

    def onclick(self, event):
        """On left mouse click without tool selected, draw new data point."""
        toolbar = plt.get_current_fig_manager().toolbar
        if (event.button==1 and toolbar.mode==''):
            if event.inaxes is not None:
                event.inaxes.plot(event.xdata,event.ydata,'x',color='r',ms=10, picker=5,label='data_pnt')
                plt.draw()

    def onpick(self, event):
        """On right mouse click on a data point, remove it."""
        if event.mouseevent.button==3:
            if hasattr(event.artist,'get_label') and (event.artist.get_label() in ('data_pnt', 'fit', 'fit_min')):
                event.artist.remove()
                plt.draw()

    def ontype(self, event):
        """On enter, fit around selected points. On 'w' write to file."""
        # fit lines around selected points
        if event.key=='enter':
            for ax in (self.ax_data, self.ax_model):
                pnt_coord = []
                for artist in ax.get_children():
                    if hasattr(artist,'get_label') and artist.get_label()=='data_pnt':
                        pnt_coord.append(artist.get_data())
                
                pnt_coord = np.array(pnt_coord)[...,0]
                sort_array = np.argsort(pnt_coord[:,0])
                if ax == self.ax_data:
                    x_data, y_data = pnt_coord[sort_array].T
                elif ax == self.ax_model:
                    x_model,y_model = pnt_coord[sort_array].T

            if len(x_data) is not len(x_model):
                raise ValueError('Number of data points = {} '
                                 'is not number of model points = {}'.format(len(x_data), len(x_model)))

            find_location_spectral_lines(x_data, y_data, self.data_wav, self.data_spec,
                                         ax=self.ax_data, nclosest=NCLOSEST_DATA, mode='gaussian')
            find_location_spectral_lines(x_model, y_model,
                                         self.telluric_wav, self.telluric_spec,
                                         ax=self.ax_model, mode='gaussian', nclosest=NCLOSEST_MODEL)

        elif event.key=='r':
            try:
                x_data, y_data, x_model, y_model = np.loadtxt(self.fpathout+'_manual_selection.txt').T
                for x1, y1, x2, y2 in zip(x_data, y_data, x_model, y_model):
                    self.ax_data.plot(x1, y1, 'x', color='r', ms=10, picker=5,label='data_pnt')
                    self.ax_model.plot(x2, y2, 'x', color='r', ms=10, picker=5,label='data_pnt')
                plt.draw()

                self.tellme('Reloaded previous selected data points from\n{}'.format(self.fpathout+'_manual_selection.txt'))
                plt.pause(1)
                self.tellme('Select center points of telluric lines in model and data\n'
                            'left mouse = select, right mouse = remove, enter = fit, w = write to file, r = reload previous selection')
            except:
                pass
                
        # write results of fit to file
        elif event.key=='w':
            for ax in (self.ax_data, self.ax_model):
                pnt_coord = []
                for artist in ax.get_children():
                    if hasattr(artist,'get_label') and artist.get_label()=='fit_min':
                        pnt_coord.append(artist.get_data())
                
                pnt_coord = np.array(pnt_coord)[...,0] #FIX THIS PART HERE! ;)
                sort_array = np.argsort(pnt_coord[:,0])
                if ax == self.ax_data:
                    x_data, y_data = pnt_coord[sort_array].T
                elif ax == self.ax_model:
                    x_model,y_model = pnt_coord[sort_array].T

            # save as a txt file
            self.tellme('')
            plt.savefig(self.fpathout+'_plot_manual_selection.png', dpi=200)
            np.savetxt(self.fpathout+'_manual_selection.txt',
                       np.c_[x_data, y_data, x_model, y_model],
                       header = 'order={}: x_data [pixel], y_data [pixel], x_model [pixel], y_model [pixel]'.format(self.norder))

            self.tellme('Saved selected data points as\n{}'.format(self.fpathout+'_manual_selection.txt'))

            plt.pause(1)
            plt.clf()
            plt.draw()
            
            self.fig.canvas.mpl_disconnect(self.cid1)
            self.fig.canvas.mpl_disconnect(self.cid2)
            self.fig.canvas.mpl_disconnect(self.cid3)
            
            ax_polyfit = self.fig.add_subplot(111)
            data_wavsolution, pnt_wavsolution, coefs_wavsolution, ax_polyfit = \
            plot_wavcal_polyfit(x_data, x_model,self.data_wav, self.norder, self.polydeg,ax_polyfit)
            
            self.tellme('Press any buttom to save and apply wavelength solution')
            plt.waitforbuttonpress()
            
            self.tellme('')
            plt.savefig(self.fpathout+'_plot_wavsolution.png', dpi=200)
            np.savetxt(self.fpathout+'_wavsolution.txt',
                       np.c_[self.data_wav, data_wavsolution],
                       header = 'order={}: data_wav [pixel], data_wavsolution [micron]'.format(self.norder))
            self.tellme('Saved wavelength solution as\n{}'.format(self.fpathout+'_wavsolution.txt'))
            plt.pause(1)
            
            plt.clf()
            plt.draw()
            
            # data subplot
            self.ax_data = self.fig.add_subplot(211)
            self.ax_model = self.fig.add_subplot(212)
        
            self.ax_data.set_title('Data spectrum calibrated (order = {})'.format(self.norder), size=12)
            self.ax_data.set_xlabel('Wavelength [micron]',size=12)
            self.ax_data.set_ylabel('Relative flux', size=12)
            self.ax_data.plot(data_wavsolution, self.data_spec, marker='x', ms=1.25, color='k')
            
            # show only within unmasked area
            leftbound = np.min(data_wavsolution[~np.isnan(self.data_spec)])
            rightbound = np.max(data_wavsolution[~np.isnan(self.data_spec)])
            self.ax_data.set_xlim(data_wavsolution[0], data_wavsolution[-1])
            
            n = np.arange(1, len(y_data)+1)
            self.ax_data.scatter(pnt_wavsolution, y_data, color=TABLEAU20[6], marker=3)
            for i, txt in enumerate(n):
                self.ax_data.annotate(txt, (pnt_wavsolution[i], y_data[i]-0.1), color=TABLEAU20[6], ha='center')
                
            n = np.arange(1, len(y_model)+1)
            self.ax_model.scatter(x_model, y_model, color=TABLEAU20[6], marker=3)
            for i, txt in enumerate(n):
                self.ax_model.annotate(txt, (x_model[i], y_model[i]-0.05), color=TABLEAU20[6], ha='center')
            
            new_wavrange = np.logical_and(self.telluric_wav >= data_wavsolution[0],
                                          self.telluric_wav <= data_wavsolution[-1])

            # telluric model subplot
            self.ax_model.set_title('Telluric model (ATRAN)'.format(self.norder), size=12)
            self.ax_model.plot(self.telluric_wav[new_wavrange], self.telluric_spec[new_wavrange], marker='x', ms=1.25, color='k')
            self.ax_model.set_xlabel('Wavelength [micron]', size=12)
            self.ax_model.set_ylabel('Relative flux', size=12)
            self.ax_model.set_xlim(data_wavsolution[0], data_wavsolution[-1])
            
            self.tellme('Result of wavelength calibration. Press any buttom to close and save wavelength solution.')
            plt.waitforbuttonpress()
            
            self.tellme('')
            plt.savefig(self.fpathout+'_plot_wavsolution_lines.png', dpi=200)
            np.savetxt(self.fpathout+'_wavsolution_lines.txt',
                       np.c_[np.arange(1, len(pnt_wavsolution)+1), x_data, pnt_wavsolution],
                       header = 'order={}: # line, wavelength [pixel], wavelength [micron]'.format(self.norder))
            self.tellme('Saved line positions as\n{}'.format(self.fpathout+'_wavsolution_lines.txt'))
            plt.pause(1)
            
            np.save(self.fpathout+'_wavsolution_coefs.npy', coefs_wavsolution)
            self.tellme('Saved coefs as\n{}'.format(self.fpathout+'_wavsolution_coefs.npy'))
            plt.pause(1)
            plt.close()
            
    def __call__(self):
        """"""
        self.fig = plt.figure()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        # create two windows
        self.tellme('Select center points of telluric lines in model and data\n'
               'left mouse = select, right mouse = remove, enter = fit, w = write to file, r = reload previous selection')
        self.ax_data = self.fig.add_subplot(211)
        self.ax_model = self.fig.add_subplot(212)

        # data subplot
        self.ax_data.set_title('Data spectrum (order = {})'.format(self.norder), size=12)
        self.ax_data.set_xlabel('X [pixel]',size=12)
        self.ax_data.set_ylabel('Relative flux', size=12)
        self.ax_data.plot(self.data_wav, self.data_spec, marker='x', ms=1.25, color='k')

        # telluric model subplot
        self.ax_model.set_title('Telluric model (ATRAN)'.format(self.norder), size=12)
        self.ax_model.plot(self.telluric_wav, self.telluric_spec, marker='x', ms=1.25, color='k')
        self.ax_model.set_xlabel('Wavelength [micron]', size=12)
        self.ax_model.set_ylabel('Relative flux', size=12)

        # connect the different functions to the different events
        self.cid1 = self.fig.canvas.mpl_connect('key_press_event', self.ontype)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid3 = self.fig.canvas.mpl_connect('pick_event', self.onpick)
        
        
class AutoWavCalMCMCSampler:
    """"""
    def __init__(self, data_pxl, data_spec, telluric_wav, telluric_spec, wavcoefs_prior):
        """"""
        self.data_pxl = data_pxl
        self.data_spec = data_spec
        self.telluric_wav = telluric_wav
        self.telluric_spec = telluric_spec
        self.wavcoefs_prior = wavcoefs_prior # Best guess of polynomial coeficients of wavelength solution
    
    def log_prior(self, theta):
        """"""
        return 0
        #         threshold = 0.25
        #         for ncoef, coef in enumerate(theta):
        #             t = np.abs((self.wavcoefs_prior[ncoef] - coef)/self.wavcoefs_prior[ncoef])
        #             if t > threshold:
        #                 return 0
        #         return -np.inf

    def log_probability(self, theta):
        """"""
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + logL_zucker(theta)
    
    def logL_zucker(self, theta):
        """"""
        # Trial wavelength solution
        data_wav = polyval(theta, self.data_pxl)
        
        # Calculate likelihood of this trial wavelength solution
        telluric_spec_interp = np.interp(x=data_wav, xp=self.telluric_wav, fp=self.telluric_spec)
        cc = np.corrcoef(self.data_spec, telluric_spec_interp)[0,1]
        N = len(self.data_spec)
        logL = -0.5*N*np.log(1-cc**2)
        #print(theta, logL)
        if not (np.isnan(logL) or np.any(~np.isfinite(theta))):
            return logL
        else:
            return 0.
    
    def run(self, theta_prior, nwalkers=8, nsteps=1e2, use_pool=False, dtheta=1e-5):
        """"""
        ndim = len(theta_prior)
        pos = theta_prior + tuple(dtheta * t for t in theta_prior) * np.random.randn(nwalkers, ndim)
        if use_pool:
            with Pool() as p:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logL_zucker, pool=p)
                try:
                    sampler.run_mcmc(pos, nsteps, progress=True)
                except:
                    return None
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logL_zucker)
            try:
                sampler.run_mcmc(pos, nsteps, progress=True)
            except:
                return None
        return sampler
    
def plot_walkers(samples):
    """"""
    ndim = samples.shape[-1]
    labels = [f'c{ndim-i-1}' for i in range(samples.shape[-1])]
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    
    #labels = ["RH", "Resolution / 10,000"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i], size=15)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step number", size=15)
    return fig, axes

def plot_corner(flat_samples):
    ndim = flat_samples.shape[-1]
    labels = [f'c{ndim-i-1}' for i in range(flat_samples.shape[-1])]
    fig = corner.corner(
        flat_samples, labels=labels
    )
    axes = fig.get_axes()
    return fig, axes

def get_result_mcmc(flat_samples, verbose=False):
    params = {}
    #labels = ['RH', 'R']
    ndim = flat_samples.shape[-1]
    for i in range(flat_samples.shape[-1]):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        
        if verbose:
            txt = "\mathrm{{{3}}} = {0:e}_{{-{1:e}}}^{{{2:e}}}"
            txt = txt.format(mcmc[1], q[0], q[1], f'c{ndim-i-1}')
            display(Math(txt))
        
        params[f'c{ndim-i-1}'] = (mcmc[1], q[0], q[1])
    return params