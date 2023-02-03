import json
import os
import pickle
import sys


from astropy.io import fits
import corner
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib import colors
import numpy as np
from numpy.core.multiarray import interp as compiled_interp
import pymultinest
from tqdm import tqdm
import scipy.stats


from aries.constants import TABLEAU20
from aries.cleanspec import butter_bandpass, butter_bandpass_filter
from aries.cleanspec import SpectralOrder, TemplateOrder
from aries.crosscorrelate import apply_rv, calc_rv_planet, plot_detection_matrix


C = 3.0e8  # speed of light in m/s


def fancy_corner_plot(
        prefix,
        n_params,
        ndim,
        labels,
        expected_values,
        threshold=5e-5):
    """Plot Pymultinest result using a modified version of corner.py."""
    # Load PyMultinest Result
    a = pymultinest.Analyzer(
        n_params=n_params,
        outputfiles_basename=prefix,
        verbose=False)
    data = a.get_data()[:, 2:]
    # convert dKp (m/s) to Kp in km/s
    data[:, 0] = data[:, 0] / 1e3 + expected_values[1]
    # convert dvsys (m/s) to vsys in km/s
    data[:, 1] = data[:, 1] / 1e3 + expected_values[0]
    data[:, 0], data[:, 1] = data[:, 1].copy(), data[:, 0].copy()

    # Mask points with small weights
    weights = a.get_data()[:, 0]
    mask = weights > threshold
    figure = corner.corner(
        data[mask],
        labels=labels,
        quantiles=[
            0.16,
            0.5,
            0.84],
        show_titles=True,
        color=TABLEAU20[0],
        label_kwargs={
            'fontsize': 12},
        title_kwargs={
            'fontsize': 12},
        fill_contours=True,
        plot_density=False,
        plot_contours=True,
        plot_points=False,
        contour_kwargs={
            'linewidths': 0},
        hist_kwargs={
            'color': TABLEAU20[0],
            'histtype': 'stepfilled',
            'alpha': 0.2})

    # This is the empirical mean of the sample:
    mean_values = np.mean(data[mask], axis=0)

    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))

    # Color scheme
    c1 = 'gray'
    c2 = TABLEAU20[0]

    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(expected_values[i], color=c1, ls=':')
        ax.axvline(mean_values[i], color=c2)

    # Loop over the marginal
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(expected_values[xi], color=c1, ls=':')
            ax.axvline(mean_values[xi], color=c2)
            ax.axhline(expected_values[yi], color=c1, ls=':')
            ax.axhline(mean_values[yi], color=c2)
            ax.scatter(
                mean_values[xi],
                mean_values[yi],
                marker='s',
                facecolors=c2,
                edgecolors=c2,
                s=50)

    return figure

def load_maxaposterior(f):
    """Load maximum a posterior parameters PyMultinest."""
    with open(f, 'r') as f:
        data = json.load(f)
        params = data["modes"][0]["maximum a posterior"]
    return params


# class BrogiLineBayesianFramework:
#     def __init__(self, spectralorders, templateorders, planetary_system,
#                  params = ['dKp', 'dvsys', 'log(a)'],
#                  priors=((-50e3, 50e3), (-50e3, 50e3), (-2,2)),
#                  subtract_mean=True):
#         """"""
#         self.observed_spec_orders = []
#         self.observed_wavs = []
#         self.template_spec_orders = []
#         self.template_wavs = []

#         bad_frames = np.all(spectralorders[0].mask, axis=1)
#         for norder, (so, to) in enumerate(zip(spectralorders, templateorders)):
#             #  Initialise Observed Spectral Orders
#             bad_wavelengths = np.array(so.mask[0,:], 'bool')
#             data = so.data[~bad_frames, :][:,~bad_wavelengths]
#             self.observed_spec_orders.append(so.data[~bad_frames, :][:,~bad_wavelengths])
#             self.observed_wavs.append(so.wavsolution[1][~bad_wavelengths])

#             #  Initialise Template Spectral Orders
#             self.template_wavs.append(to.wavegrid)
#             self.template_spec_orders.append(to.data.T[0,:])

#         self.phases = spectralorders[0].phase[~bad_frames] #  Same for all orders
#         self.vbarys = spectralorders[0].vbary[~bad_frames]

#         self.norders = len(spectralorders)
#         self.nframes = len(self.phases)
#         self.nwavs = np.array([spectral_order.shape[1] \
#                  for spectral_order in self.observed_spec_orders])

#         self.observed_vars = [np.std(spectral_order, axis=1) \
#                          for spectral_order in self.observed_spec_orders]

#         self.kp = planetary_system['kp']
#         self.vsys = planetary_system['vsys']
        
#         self.priors = priors
#         self.params = params
#         self.nparams = len(priors)
        
#         if subtract_mean:
#             for norder in range(self.norders):
#                 self.observed_spec_orders[norder] -= self.observed_spec_orders[norder].mean(axis=1)[:,np.newaxis]
#             self.template_spec_orders[norder] -= self.template_spec_orders[norder].mean()
    
#     def prior(self, cube, ndim, nparams):
#         """"""
#         cube[0] = cube[0] * (self.priors[0][1] -
#                              self.priors[0][0]) + self.priors[0][0]
#         cube[1] = cube[1] * (self.priors[1][1] -
#                              self.priors[1][0]) + self.priors[1][0]
#         cube[2] = cube[2] * (self.priors[2][1] -
#                              self.priors[2][0]) + self.priors[2][0]
        
#     def loglike(self, cube, ndim, nparams):
#         """"""
#         dvsys = cube[0]
#         dkp = cube[1]
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
#             model_var = np.std(model, axis=1)
#             R = (1./self.nwavs[norder]) * np.sum(model * tuple(self.observed_spec_orders)[norder], axis=1)
#             logL_values.append(np.sum(-0.5 * self.nwavs[norder] * np.log(model_var + R + self.observed_vars[norder])))
#         return np.sum(logL_values)
    
#     def run(self, dirout='out/', **pymultinest_kwargs):
#         """Run pymultinest routine."""
#         if not os.path.exists(dirout):
#             os.mkdir(dirout)

#         pymultinest.run(
#             self.loglike,
#             self.prior,
#             self.nparams,
#             outputfiles_basename=dirout,
#             **pymultinest_kwargs)
#         json.dump(
#             self.params,
#             open(
#                 f'{dirout}params.json',
#                 'w'))  # save parameter names
#         # create a minimal corner plot
#         os.system(f"multinest_marginals.py {dirout}")

class BrogiLineBayesianFramework_1DTemplate:
    def __init__(self, spectralorders, templateorders, planetary_system,
                 params = ['dkp', 'dvsys', 'log(a)'],
                 priors=((-50e3, 50e3), (-50e3, 50e3), (-2,2)),
                 subtract_mean=True):
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
    
    @property
    def ntotal(self):
        return sum(self.nwavs[m]*self.nframes[m] for m in range(self.norders))
    
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
        
class BrogiLineBayesianFramework_2DTemplate:
    def __init__(self, spectralorders, templateorders, planetary_system,
                 params = ['dkp', 'dvsys', 'log(a)'],
                 priors=((-50e3, 50e3), (-50e3, 50e3), (-2,2)),
                 subtract_mean=True):
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
            self.template_spec_orders.append(to.data.T[~bad_frames,:])
        
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
                self.template_spec_orders[norder] -= self.template_spec_orders[norder].mean(axis=1)[:,np.newaxis]
    
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
                          fp=template_spec_scaled[nframe,:]) for nframe in range(self.nframes[norder])))

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