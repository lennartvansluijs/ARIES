import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import numpy as np
from astropy.io import fits
import json
import pymultinest
import corner
from aries.constants import TABLEAU20
from aries.cleanspec import butter_bandpass, butter_bandpass_filter
from tqdm import tqdm
import pickle
import json
import scipy.stats
import pymultinest
from matplotlib import colors
from aries.cleanspec import SpectralOrder, TemplateOrder
from numpy.core.multiarray import interp as compiled_interp


from aries.crosscorrelate import apply_rv, calc_rv_planet, plot_detection_matrix

def fancy_corner_plot(prefix, n_params, ndim, labels, expected_values):
    """Plot Pymultinest result using a modified version of corner.py."""
    # Load data
    a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=prefix, verbose=False)

    # Get data
    data = a.get_data()[:,2:]
    data[:,0] = data[:,0]/1e3 + expected_values[1] # convert dKp (m/s) to Kp (km/s)
    data[:,1] = data[:,1]/1e3 + expected_values[0] # convert dvsys (m/s) to vsys (km/s)
    data[:,0], data[:,1] = data[:,1].copy(), data[:,0].copy()

    # Mask points with small weights
    weights = a.get_data()[:,0]
    mask = weights > 5e-5

    figure = corner.corner(data[mask], labels=labels,
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, color=TABLEAU20[0], label_kwargs={'fontsize':12}, title_kwargs={'fontsize':12},
                          fill_contours=True, plot_density=False, plot_contours=True, plot_points=False, contour_kwargs={'linewidths':0},
                          hist_kwargs={'color':TABLEAU20[0], 'histtype':'stepfilled', 'alpha':0.2})
    
    # This is the empirical mean of the sample:
    mean_values = np.mean(data[mask], axis=0)

    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))

    # Color scheme
    c1='gray'
    c2=TABLEAU20[0]
    
    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(expected_values[i], color=c1, ls=':')
        ax.axvline(mean_values[i], color=c2)

    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(expected_values[xi], color=c1, ls=':')
            ax.axvline(mean_values[xi], color=c2)
            ax.axhline(expected_values[yi], color=c1, ls=':')
            ax.axhline(mean_values[yi], color=c2)
            ax.scatter(mean_values[xi], mean_values[yi], marker='s', facecolors=c2, edgecolors=c2, s=50)
    
    return figure
            
def calc_chi2(data, model, error, alpha=1.):
    """Calculate the chi square value."""
    return np.sum( (data-alpha*model)**2 / error**2 )

class HRCCSRetrieval:
    """High Resolution Cross Correlation Spectroscopy Retrieval."""
    def __init__(self, spectralorders,
                 template_wav, template_spec,
                 vsys, vbary, kp, **kwargs):
        """"""
        # subtract mean in order to normalise each spectrum
        # (see Brogi & Line 2019 why this is necessary)
        self.spectralorders = spectralorders
        for so in self.spectralorders:
            so.data -= so.data.mean(axis=1)[:,np.newaxis]
        self.masks = [so.mask[0,:].astype('bool') for so in self.spectralorders] # use 1d boolean masks
        
        # same for the template synthetic spectrum
        self.template_wav = template_wav
        self.template_spec = template_spec - template_spec.mean()
        
        # define some additional parameters
        self.norders = len(spectralorders)
        self.data_phase = spectralorders[0].phase # same phase for all spectral orders
        self.vsys = vsys
        self.kp = kp
        self.vbary = vbary
        
        self.kp_range = kwargs.pop('kp_range', (-50e3, 50e3))
        self.vsys_range = kwargs.pop('vsys_range', (-50e3, 50e3))
        self.a_range = kwargs.pop('a_range', (0., 10.))
        self.parameters = kwargs.pop('parameters', ["dkp", "dvsys", "a"])
        self.n_params = len(self.parameters)
        self.loglike_method = kwargs.pop('loglike_method', 'bl19') # or choose 'gb20' for Gibson et al. 2020
        self.model_kwargs = kwargs.pop('model_kwargs', {})
        
        if self.model_kwargs['apply_hpf']:
            self.hpf_bandpass = butter_bandpass(freq_cutoff=self.model_kwargs['hpf_fc']) # define highpass filter bandpass once

    def eval_models(self, dvsys, dkp, a, apply_hpf=False, hpf_fc=None):
        """Calculate the synthetic model spectrum for all spectral orders."""
        # shift template to the planet's rv
        rv_planet = calc_rv_planet(self.vbary, self.vsys+dvsys, self.kp+dkp, self.data_phase)
        
        models = []
        for so in self.spectralorders:
            model = np.zeros(so.data.shape)
            for n, rv in enumerate(rv_planet):
                model[n,:] = np.interp(x=so.wavsolution[1],
                                       xp=apply_rv(self.template_wav, rv),
                                       fp=self.template_spec) # template spec may need to become a data cube itself to account for different observed spectral resolutions...
            if apply_hpf:
                models.append(a * butter_bandpass_filter(model, *self.hpf_bandpass))
            else:
                models.append(a * model)
        return models
        
    def prior(self, cube, ndim, nparams):
        """"""
        cube[0] = cube[0] * (self.kp_range[1]-self.kp_range[0]) + self.kp_range[0]
        cube[1] = cube[1] * (self.vsys_range[1]-self.vsys_range[0]) + self.vsys_range[0]
        cube[2] = cube[2] * (self.a_range[1]-self.a_range[0]) + self.a_range[0]

    def loglike(self, cube, ndim, nparams):
        """"""
        # use Brogi & Line 2019 log L
        if self.loglike_method == 'bl19':
            loglike_all = []
            models = self.eval_models(dkp=cube[0], dvsys=cube[1], a=cube[2], **self.model_kwargs)
            for model, so, mask in zip(models, self.spectralorders, self.masks):      
                nobs = so.data.shape[0]
                N = sum(~mask)
                for j in range(nobs):
                    sf2 = (1./N)*np.sum(so.data[j,~mask]**2)
                    sg2 = (1./N)*np.sum(model[j,~mask]**2)
                    R = (1./N)*np.sum(so.data[j,~mask]*model[j,~mask])
                    logL = -1*(N/2.)*np.log(sf2 - 2*R + sg2)
                    loglike_all.append(logL)
            loglike = sum(loglike_all)
            
        # use Gibson et al. 2020 log L
        elif self.loglike_method == 'gb20':
            loglike_all = []
            models = self.eval_models(dkp=cube[0], dvsys=cube[1], a=cube[2], **self.model_kwargs)
            for model, so, mask in zip(models, self.spectralorders, self.masks):      
                nobs, M = so.data.shape
                N = mask.sum()
                for j in range(nobs):
                    chi2 = calc_chi2(data=so.data[j,~mask], model=model[j,~mask], error=so.error[j,~mask], alpha=cube[2])
                    lnZ = -1*(N/2.)*np.log(chi2/N)
                    loglike_all.append(lnZ)
            loglike = sum(loglike_all)
            
        # method not implemented
        else:
            raise ValueError(f'Invalid log L method: {self.loglike_method}.')
            
        return loglike
    
    def run_multinest(self, dirout='out/', **kwargs):
        """Run pymultinest routine."""
        if not os.path.exists(dirout):
            os.mkdir(dirout)
                
        pymultinest.run(self.loglike, self.prior, self.n_params, outputfiles_basename=dirout, **kwargs) # run multinest routine
        json.dump(self.parameters, open(f'{dirout}params.json', 'w')) # save parameter names
        os.system(f"multinest_marginals.py {dirout}") # create a minimal corner plot
    
    def plot_multinest(self, dirin='out/'):
        """Plot marginals of output multinest routine."""
        fig = fancy_corner_plot(prefix=dirin, n_params=self.n_params, ndim=self.n_params,
                          expected_values=(self.vsys/1e3, self.kp/1e3, 1.),
                          labels=[r"$v_{\rm{sys}}$ [km/s]",r"$K_{\rm{p}}$ [km/s]", r"$\alpha$"])
        
        # save figure
        f = os.path.join(dirin, 'fancy_corner')
        plt.savefig(f+'.png', dpi=200)
        plt.savefig(f+'.pdf')
        plt.close()
        
    def run_gridsearch(self, dvsys_all, dkp_all, a=1., mode='default', dirout='out/'):
        """"""
        if not os.path.exists(dirout):
            os.mkdir(dirout)
            
        if mode is 'default':
            loglike_grid = np.zeros(shape=(len(dvsys_all), len(dkp_all)))
            with tqdm(total= loglike_grid.size, desc='log L grid search') as pbar:
                for i, dvsys in enumerate(dvsys_all):
                    for j, dkp in enumerate(dkp_all):
                        cube = (dkp, dvsys, a)
                        loglike_grid[i,j] = self.loglike(cube, ndim=2, nparams=2)
                        pbar.update()
        elif mode is 'null':
            loglike_grid = np.zeros(shape=(len(dvsys_all), len(dkp_all)))
            cube = (0.,0.,0.)
            loglike_grid[:] = self.loglike(cube, ndim=2, nparams=2)
        else:
            raise ValueError(f'Invalid mode: {mode}.')
        
        # save result of grid search
        fpathout = os.path.join(dirout, 'loglike.fits')
        fits.writeto(fpathout, loglike_grid, overwrite=True)
        
        fpathout = os.path.join(dirout, 'grid.txt')
        header = f'center: (vsys, kp) = ({self.vsys, self.kp}) \n'         'grid: dvsys_all (m/s) | dkp_all (m/s)'
        np.savetxt(fpathout, np.c_[dvsys_all, dkp_all],
                   header=header, delimiter=',')
        
        center = (self.vsys, self.kp)
        with open(f'{dirout}center.pickle', 'wb') as f:
            pickle.dump(center, f)
    
    def plot_gridsearch(self, dirin='out/', cmap='Blues_r',
                        unit='log(L)', title='Brogi & Line 2019 log(L)-Matrix',
                        interp_method='bicubic', sf=7.5, ncolors=10):
        """"""
        # load files
        loglike_grid = fits.getdata(os.path.join(dirin, 'loglike.fits'))
        data = np.loadtxt(os.path.join(dirin, 'grid.txt'), dtype='str', delimiter=',').astype('float')
        dvsys_all = data[:,0]
        dkp_all = data[:,1]
        with open(f'{dirin}center.pickle', 'rb') as f:
            vsys, kp = pickle.load(f)
        
        # Plot result
        if self.loglike_method == 'bl19':
            text = 'Brogi & Line 2019 log L Matrix'
        elif self.loglike_method == 'gb20':
            text = 'Gibson et al. 2020 log L Matrix'
        else:
            raise ValueError(f'Unknown loglike method: {self.loglike_method}')
        
        axes = plot_detection_matrix(data=loglike_grid, dkp_all=dkp_all, dvsys_all=dvsys_all,
                                     kp=kp, vsys=vsys, title=text, mode='loglike')
        
        # save figure
        f = os.path.join(dirin, 'loglike_grid')
        plt.savefig(f+'.png', dpi=200)
        plt.savefig(f+'.pdf')
        plt.close()

def load_maxaposterior(f):
    """Load maximum a posterior parameters found by pymultinest from stats.json file."""
    with open(f, 'r') as f:
        data = json.load(f)
        params = data["modes"][0]["maximum a posterior"]
    return params


def butter_bandpass(freq_cutoff, **kwargs):
    """Return high pass filtered data."""
    filter_order = kwargs.pop('N', 6)
    b, a = signal.butter(N=filter_order, Wn=freq_cutoff, btype='highpass', output='ba', fs=1, **kwargs)
    return b, a

def fast_butter_bandpass_filter(data, b, a):
    """Butterworth highpassfilter."""
    data_f = np.zeros(data.shape)
    for n in range(data.shape[0]):
        data_f[n,:] = signal.filtfilt(b, a, data[n,:])   
    return data_f

class FastHRCCSRetrieval:
    """High Resolution Cross Correlation Spectroscopy Retrieval."""
    def __init__(self, spectralorders, templateorders,
                 vsys, vbary, kp, **kwargs):
        """"""
        # subtract mean in order to normalise each spectrum
        # (see Brogi & Line 2019 why this is necessary)
        self.spectralorders = [SpectralOrder(data=so.data, mask=so.mask, norder=so.norder, wavsolution=so.wavsolution, target=so.target, phase=so.phase, time=so.time) for so in spectralorders]
        self.templateorders = [TemplateOrder(data=to.data, wav=to.wavegrid, norder=to.norder, R_values=to.R_values, targetname=to.targetname) for to in templateorders]
        
        # masked wavelengths
        self.masks = [np.all(so.mask, axis=0) for so in self.spectralorders]
        
        # Remove bad frames/masked phases
        for so, to in zip(self.spectralorders, self.templateorders):
            phase_mask = np.all(so.mask, axis=1)
            
            so.data = so.data[~phase_mask,:]
            so.nobs = so.data.shape[0]
            so.mask = so.mask[~phase_mask,:]
            so.phase = so.phase[~phase_mask]
            so.time = so.time[~phase_mask]
            so.vbary = vbary[~phase_mask]

            to.data = to.data[:,~phase_mask]
            
        for so, to in zip(self.spectralorders, self.templateorders):
            so.data -= so.data.mean(axis=1)[:,np.newaxis]
            to.data -= to.data.mean(axis=0)[np.newaxis,:]
        
        # define some additional parameters
        self.norders = len(spectralorders)
        self.vsys = vsys
        self.kp = kp
        
        self.kp_range = kwargs.pop('kp_range', (-50e3, 50e3))
        self.vsys_range = kwargs.pop('vsys_range', (-50e3, 50e3))
        self.a_range = kwargs.pop('a_range', (0, 10))
        self.parameters = kwargs.pop('parameters', ["dkp", "dvsys", "a"])
        self.n_params = len(self.parameters)

    def eval_models(self, dvsys, dkp, a):
        """Calculate the synthetic model spectrum for all spectral orders."""
        # shift template to the planet's rv
        #rv_planet = calc_rv_planet(self.vbary, self.vsys+dvsys, self.kp+dkp, self.data_phase)
        models = []
        for so, to in zip(self.spectralorders, self.templateorders):
            rv_planet = -so.vbary + (self.vsys+dvsys) + (self.kp+dkp)*np.sin(so.phase*2*np.pi)
            model = np.zeros(so.data.shape)
            for n, rv in enumerate(rv_planet):
                model[n,:] = np.interp(x=so.wavsolution[1],
                                       xp=apply_rv(to.wavegrid, rv),
                                       fp=to.data[:,n])
            models.append(a * model) # no high pass filter in fast mode...
        return models
        
    def prior(self, cube, ndim, nparams):
        """"""
        cube[0] = cube[0] * (self.kp_range[1]-self.kp_range[0]) + self.kp_range[0]
        cube[1] = cube[1] * (self.vsys_range[1]-self.vsys_range[0]) + self.vsys_range[0]
        cube[2] = cube[2] * (self.a_range[1]-self.a_range[0]) + self.a_range[0]

    def loglike(self, cube, ndim, nparams):
        """"""
        # use Brogi & Line 2019 log L
        loglike_all = []
        models = self.eval_models(dkp=cube[0], dvsys=cube[1], a=cube[2])
        for model, so, mask in zip(models, self.spectralorders, self.masks):      
            nobs = so.data.shape[0]
            N = sum(~mask)
            for j in range(nobs):
                sf2 = (1./N)*np.sum(so.data[j,~mask]**2)
                sg2 = (1./N)*np.sum(model[j,~mask]**2)
                R = (1./N)*np.sum(so.data[j,~mask]*model[j,~mask])
                logL = -1*(N/2.)*np.log(sf2 - 2*R + sg2)
                loglike_all.append(logL)  
        return sum(loglike_all)
    
    def loglike_all_orders(self, cube, ndim, nparams):
        """"""
        # use Brogi & Line 2019 log L
        loglike_all_orders = np.zeros(self.norders) + np.nan
        models = self.eval_models(dkp=cube[0], dvsys=cube[1], a=cube[2])
        for order, (model, so, mask) in enumerate(zip(models, self.spectralorders, self.masks)):      
            nobs = so.data.shape[0]
            N = sum(~mask)
            loglike_all = []
            for j in range(nobs):
                sf2 = (1./N)*np.sum(so.data[j,~mask]**2)
                sg2 = (1./N)*np.sum(model[j,~mask]**2)
                R = (1./N)*np.sum(so.data[j,~mask]*model[j,~mask])
                logL = -1*(N/2.)*np.log(sf2 - 2*R + sg2)
                loglike_all.append(logL)
            loglike_all_orders[order] = sum(loglike_all)
        return loglike_all_orders
    
    def run_multinest(self, dirout='out/', **kwargs):
        """Run pymultinest routine."""
        if not os.path.exists(dirout):
            os.mkdir(dirout)
                
        pymultinest.run(self.loglike, self.prior, self.n_params, outputfiles_basename=dirout, **kwargs) # run multinest routine
        json.dump(self.parameters, open(f'{dirout}params.json', 'w')) # save parameter names
        os.system(f"multinest_marginals.py {dirout}") # create a minimal corner plot
    
    def plot_multinest(self, dirin='out/'):
        """Plot marginals of output multinest routine."""
        fig = fancy_corner_plot(prefix=dirin, n_params=self.n_params, ndim=self.n_params,
                          expected_values=(self.vsys/1e3, self.kp/1e3, 1.),
                          labels=[r"$v_{\rm{sys}}$ [km/s]",r"$K_{\rm{p}}$ [km/s]", r"$\alpha$"])
        
        # save figure
        f = os.path.join(dirin, 'fancy_corner')
        plt.savefig(f+'.png', dpi=200)
        plt.savefig(f+'.pdf')
        plt.close()
        
    def run_gridsearch(self, dvsys_all, dkp_all, a=1., mode='default', dirout='out/'):
        """"""
        if not os.path.exists(dirout):
            os.mkdir(dirout)
            
        if mode is 'default':
            loglike_grid = np.zeros(shape=(len(dvsys_all), len(dkp_all)))
            with tqdm(total= loglike_grid.size, desc='log L grid search') as pbar:
                for i, dvsys in enumerate(dvsys_all):
                    for j, dkp in enumerate(dkp_all):
                        cube = (dkp, dvsys, a)
                        loglike_grid[i,j] = self.loglike(cube, ndim=2, nparams=2)
                        pbar.update()
        if mode is 'all_orders':
            loglike_grid_all_orders = np.zeros(shape=(len(dvsys_all), len(dkp_all), self.norders))
            with tqdm(total= loglike_grid_all_orders[:,:,0].size, desc='log L grid search') as pbar:
                for i, dvsys in enumerate(dvsys_all):
                    for j, dkp in enumerate(dkp_all):
                        cube = (dkp, dvsys, a)
                        loglike_grid_all_orders[i,j,:] = self.loglike_all_orders(cube, ndim=2, nparams=2)
                        pbar.update()
            loglike_grid = np.sum(loglike_grid_all_orders, axis=2)
        elif mode is 'null':
            loglike_grid = np.zeros(shape=(len(dvsys_all), len(dkp_all)))
            cube = (0.,0.,0.)
            loglike_grid[:] = self.loglike(cube, ndim=2, nparams=2)
        else:
            raise ValueError(f'Invalid mode: {mode}.')
        
        # save result of grid search
        fpathout = os.path.join(dirout, 'loglike.fits')
        fits.writeto(fpathout, loglike_grid, overwrite=True)
        
        # save result of grid search all orders
        fpathout = os.path.join(dirout, 'loglike_all_orders.fits')
        fits.writeto(fpathout, loglike_grid_all_orders, overwrite=True)
        
        fpathout = os.path.join(dirout, 'grid.txt')
        header = f'center: (vsys, kp) = ({self.vsys, self.kp}) \n'         'grid: dvsys_all (m/s) | dkp_all (m/s)'
        np.savetxt(fpathout, np.c_[dvsys_all, dkp_all],
                   header=header, delimiter=',')
        
        center = (self.vsys, self.kp)
        with open(f'{dirout}center.pickle', 'wb') as f:
            pickle.dump(center, f)
        
    
    def plot_gridsearch(self, dirin='out/', cmap='Blues_r',
                        unit='log(L)', title='Brogi & Line 2019 log(L)-Matrix',
                        interp_method='bicubic', sf=7.5, ncolors=10):
        """"""
        # load files
        loglike_grid = fits.getdata(os.path.join(dirin, 'loglike.fits'))
        data = np.loadtxt(os.path.join(dirin, 'grid.txt'), dtype='str', delimiter=',').astype('float')
        dvsys_all = data[:,0]
        dkp_all = data[:,1]
        with open(f'{dirin}center.pickle', 'rb') as f:
            vsys, kp = pickle.load(f)
        
        # Plot result
        if self.loglike_method == 'bl19':
            text = 'Brogi & Line 2019 log L Matrix'
        elif self.loglike_method == 'gb20':
            text = 'Gibson et al. 2020 log L Matrix'
        else:
            raise ValueError(f'Unknown loglike method: {self.loglike_method}')
        
        axes = plot_detection_matrix(data=loglike_grid, dkp_all=dkp_all, dvsys_all=dvsys_all,
                                     kp=kp, vsys=vsys, title=text, mode='loglike')
        
        # save figure
        f = os.path.join(dirin, 'loglike_grid')
        plt.savefig(f+'.png', dpi=200)
        plt.savefig(f+'.pdf')
        plt.close()

def load_maxaposterior(f):
    """Load maximum a posterior parameters found by pymultinest from stats.json file."""
    with open(f, 'r') as f:
        data = json.load(f)
        params = data["modes"][0]["maximum a posterior"]
    return params




class BrogiLineBayesianFramework:
    def __init__(self, spectralorders, templateorders, planetary_system,
                 params = ['dkp', 'dvsys', 'log(a)'],
                 priors=((-50e3, 50e3), (-50e3, 50e3), (-2,2)),
                 subtract_mean=True, apply_hpf_to_model=False):
        """"""
        self.observed_spec_orders = []
        self.observed_wavs = []
        self.template_spec_orders = []
        self.template_wavs = []
        self.obsdates = []
        self.phases = []
        self.vbarys = []
        self.nframes = []
        self.nwavs = []
        
        for norder, (so, to) in enumerate(zip(spectralorders, templateorders)):
            #  Initialise Observed Spectral Orders
            bad_frames = np.all(so.mask, axis=1)
            bad_wavelengths = np.array(so.mask[~bad_frames, :][0, :], 'bool')  #  Same for each column, per order
            data = so.data[~bad_frames, :][:,~bad_wavelengths]
            
            self.nframes.append(data.shape[0])
            self.nwavs.append(data.shape[1])
            self.vbarys.append(so.vbary[~bad_frames])
            self.phases.append(so.phase[~bad_frames])
            self.observed_spec_orders.append(so.data[~bad_frames, :][:,~bad_wavelengths])
            self.observed_wavs.append(so.wavsolution[1][~bad_wavelengths])
            self.obsdates.append(so.obsdate)
            
            #  Initialise Template Spectral Orders
            self.template_wavs.append(to.wavegrid)
            self.template_spec_orders.append(to.data.T[0,:])
        
        self.norders = len(spectralorders)
        self.observed_vars = [np.var(so, axis=1) for so in self.observed_spec_orders]

        self.kp = planetary_system['kp']
        self.vsys = planetary_system['vsys']
        
        self.priors = priors
        self.params = params
        self.nparams = len(priors)
        
        if subtract_mean:
            for norder in range(self.norders):
                self.observed_spec_orders[norder] -= self.observed_spec_orders[norder].mean(axis=1)[:,np.newaxis]
            self.template_spec_orders[norder] -= self.template_spec_orders[norder].mean()
    
    def prior(self, cube, ndim, nparams):
        """"""
        cube[0] = cube[0] * (self.priors[0][1] -
                             self.priors[0][0]) + self.priors[0][0]
        cube[1] = cube[1] * (self.priors[1][1] -
                             self.priors[1][0]) + self.priors[1][0]
        cube[2] = cube[2] * (self.priors[2][1] -
                             self.priors[2][0]) + self.priors[2][0]
        
    def loglike(self, cube, ndim, nparams):
        """"""
        dkp = cube[0]
        dvsys = cube[1]
        loga = cube[2]
        a = 10**(loga)

        logL_values = []
        for norder in range(self.norders):
            #  Calculate Doppler shift
            dv = -self.vbarys[norder] + (self.vsys + dvsys) + (self.kp + dkp) * np.sin(self.phases[norder] * 2 * np.pi)
            doppler_s = (1.+dv/C)  #  assumes a classical Doppler shift

            #  Apply Doppler shift to template
            template_wavs_s = np.outer(doppler_s, self.template_wavs[norder])
            template_spec_scaled = a * self.template_spec_orders[norder]
            model = np.vstack(list(
                compiled_interp(x=self.observed_wavs[norder],
                          xp=template_wavs_s[nframe],
                          fp=template_spec_scaled) for nframe in range(self.nframes[norder])))

            #  Calculate logL Brogi & Line '19
            model_var = np.var(model, axis=1)
            R = (1./self.nwavs[norder]) * np.sum(model * self.observed_spec_orders[norder], axis=1)
            logL_values.append(np.sum( -1 * (self.nwavs[norder]/2.) * np.log(model_var - 2 * R + self.observed_vars[norder])))
        return np.sum(logL_values)
    
    def run_multinest(self, dirout='out/', **pymultinest_kwargs):
        """Run pymultinest routine."""
        if not os.path.exists(dirout):
            os.mkdir(dirout)

        pymultinest.run(
            self.loglike,
            self.prior,
            self.nparams,
            outputfiles_basename=dirout,
            **pymultinest_kwargs)
        json.dump(
            self.params,
            open(
                f'{dirout}params.json',
                'w'))  # save parameter names
        # create a minimal corner plot
        os.system(f"multinest_marginals.py {dirout}")
    
    def plot_multinest(self, dirin='out/'):
        """Plot marginals of output multinest routine."""
        fig = fancy_corner_plot(prefix=dirin, n_params=self.nparams, ndim=self.nparams,
                          expected_values=(self.vsys/1e3, self.kp/1e3, 0.),
                          labels=[r"$v_{\rm{sys}}$ [km/s]",r"$K_{\rm{p}}$ [km/s]", r"$\log{a}$"])
        
        # save figure
        f = os.path.join(dirin, 'fancy_corner')
        plt.savefig(f+'.png', dpi=200)
        plt.savefig(f+'.pdf')
        plt.close()
        
    def run_gridsearch(self, dvsys_all, dkp_all, loga=0., dirout='out/'):
        """"""
        if not os.path.exists(dirout):
            os.mkdir(dirout)
            
        loglike_grid = np.zeros(shape=(len(dvsys_all), len(dkp_all)))
        with tqdm(total= loglike_grid.size, desc='log L grid search') as pbar:
            for i, dvsys in enumerate(dvsys_all):
                for j, dkp in enumerate(dkp_all):
                    cube = (dkp, dvsys, loga)
                    loglike_grid[i,j] = self.loglike(cube, ndim=2, nparams=2)
                    pbar.update()
        
        # save result of grid search
        fpathout = os.path.join(dirout, 'loglike.fits')
        fits.writeto(fpathout, loglike_grid, overwrite=True)
        
        fpathout = os.path.join(dirout, 'grid.txt')
        header = f'center: (vsys, kp) = ({self.vsys, self.kp}) \n'         'grid: dvsys_all (m/s) | dkp_all (m/s)'
        np.savetxt(fpathout, np.c_[dvsys_all, dkp_all],
                   header=header, delimiter=',')
        
        center = (self.vsys, self.kp)
        with open(f'{dirout}center.pickle', 'wb') as f:
            pickle.dump(center, f)
    
    def plot_gridsearch(self, dirin='out/', cmap='Blues_r',
                        unit='log(L)', title='Brogi & Line 2019 log(L)-Matrix',
                        interp_method='bicubic', sf=7.5, ncolors=10):
        """"""
        # load files
        loglike_grid = fits.getdata(os.path.join(dirin, 'loglike.fits'))
        data = np.loadtxt(os.path.join(dirin, 'grid.txt'), dtype='str', delimiter=',').astype('float')
        dvsys_all = data[:,0]
        dkp_all = data[:,1]
        with open(f'{dirin}center.pickle', 'rb') as f:
            vsys, kp = pickle.load(f)
        
        # Plot result
        axes = plot_detection_matrix(data=loglike_grid, dkp_all=dkp_all, dvsys_all=dvsys_all,
                                     kp=kp, vsys=vsys, title='Brogi & Line 2019 log L Matrix',
                                     mode='loglike')
        
        # save figure
        f = os.path.join(dirin, 'loglike_grid')
        plt.savefig(f+'.png', dpi=200)
        plt.savefig(f+'.pdf')
        plt.close()



    #     def loglike_full(self, cube, ndim, nparams):
    #         """"""
    #         dkp = cube[0]
    #         dvsys = cube[1]
    #         loga = cube[2]
    #         a = 10**(loga)

    #         #  Calculate Doppler shift
    #         dv = -self.vbarys + (self.vsys + dvsys) + (self.kp + dkp) * np.sin(self.phases * 2 * np.pi)
    #         doppler_s = (1.+dv/C)  #  assumes a classical Doppler shift

    #         logL_values = []
    #         for norder in range(self.norders):

    #             #  Apply Doppler shift to template
    #             template_wavs_s = np.outer(doppler_s, self.template_wavs[norder])
    #             template_spec_scaled = a * self.template_spec_orders[norder]
    #             model = np.vstack(list(
    #                 compiled_interp(x=self.observed_wavs[norder],
    #                           xp=template_wavs_s[nframe],
    #                           fp=template_spec_scaled) for nframe in range(self.nframes)))

    #             #  Calculate logL Brogi & Line '19
    #             model_var = np.var(model, axis=1)
    #             R = (1./self.nwavs[norder]) * np.sum(model * self.observed_spec_orders[norder], axis=1)
    #             logL_values.append(-1 * (self.nwavs[norder]/2.) * np.log(model_var - 2 * R + self.observed_vars[norder]))
    #         return logL_values