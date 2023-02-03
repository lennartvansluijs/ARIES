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

from .crosscorrelate import apply_rv, calc_rv_planet, plot_detection_matrix

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
            
def calc_chi2(data, model, error, alpha=1.):
    """Calculate the chi square value."""
    return np.sum( (data-alpha*model)**2 / error**2 )

class HRCCSRetrieval:
    """High Resolution Cross Correlation Spectroscopy Retrieval."""
    def __init__(self, spectralorders,
                 template_wav, template_spec,
                 vsys, kp, **kwargs):
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
        self.vsys = vsys
        self.kp = kp
        
        self.kp_range = kwargs.pop('kp_range', (-50e3, 50e3))
        self.vsys_range = kwargs.pop('vsys_range', (-50e3, 50e3))
        self.a_range = kwargs.pop('a_range', (0, 2))
        self.parameters = kwargs.pop('parameters', ["dkp", "dvsys", "a"])
        self.n_params = len(self.parameters)
        self.loglike_method = kwargs.pop('loglike_method', 'bl19') # or choose 'gb20' for Gibson et al. 2020
        self.model_kwargs = kwargs.pop('model_kwargs', {})
        
        if self.model_kwargs['apply_hpf']:
            self.hpf_bandpass = butter_bandpass(freq_cutoff=self.model_kwargs['hpf_fc']) # define highpass filter bandpass once

    def eval_models(self, dvsys, dkp, a, apply_hpf=False, hpf_fc=None):
        """Calculate the synthetic model spectrum for all spectral orders."""
        # shift template to the planet's rv
        models = []
        for so in self.spectralorders:
            rv_planet = calc_rv_planet(so.vbary, self.vsys+dvsys, self.kp+dkp, so.phase)
            model = np.zeros(so.data.shape)
            for n, rv in enumerate(rv_planet):
                model[n,:] = np.interp(x=so.wavsolution[1],
                                       xp=apply_rv(self.template_wav, rv),
                                       fp=self.template_spec)
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
        fancy_corner_plot(prefix=dirin, n_params=self.n_params, ndim=self.n_params,
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