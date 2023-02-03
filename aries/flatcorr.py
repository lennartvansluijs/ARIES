import os
from collections import namedtuple

import numpy as np
import numpy.polynomial.polynomial as polynomial
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy
from scipy.stats import norm
from scipy.signal import medfilt2d

from .preprocessing import robust_polyfit
from .preprocessing import is_flat
from .preprocessing import plot_image

from .traces import EchelleTraces, EchelleImageTransformer

import warnings
import matplotlib
from matplotlib import gridspec

def get_flats_fnames(dirin):
    """Return list of flats and corresponding traces."""
    STEM = slice(0, -5, 1) #no .fits extension
    fnames = [fname[STEM] for fname in os.listdir(dirin) if is_flat(fname) and fname.endswith('.fits')]
    return fnames

def load_traces(dirin, fnames):
    traces = []
    for fname in fnames:
        traces.append(EchelleTraces.load(os.path.join(dirin, fname+'.pkl')))
    return traces

def load_flats(dirin, fnames):
    flats = []
    for fname in fnames:
        flats.append(fits.getdata(os.path.join(dirin, fname+'.fits')))
    return flats

def fit_illumination_model(img, traces, yoffset, polydegree=7, aperture=25, sigma=3., return_badpixelmap=False):
    """Return illumination model of a dewarped image."""
    x = np.arange(img.shape[0])
    y0_traces = np.array([trace.y[0] for trace in traces.traces])
    y0_traces_dewarped = y0_traces - yoffset
    
    illumination_model = np.full(img.shape, np.nan)
    badpixelmap = np.full(img.shape, np.nan)
    
    for y0 in y0_traces_dewarped:
        aperture_window = np.arange(int(y0)-int(aperture/2.),
                                    int(y0)+int(aperture/2.)) - 1
        for i in aperture_window:
            row = img[i, :]
            is_valid = ~np.isnan(row)
            bestfit, outliers = robust_polyfit(x[is_valid],
                                             row[is_valid],
                                             deg=polydegree,
                                             return_outliers=True, sigma=sigma)
            illumination_model[i,:][is_valid] = bestfit
            badpixelmap[i,:][is_valid] = outliers
    if return_badpixelmap:
        return illumination_model, badpixelmap
    else:
        return illumination_model
    
def get_power_spectrum(signal):
    """Return Fourier power spectrum of signal and corresponding frequencies."""
    ft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))
    power = np.abs(ft**2)
    positive_freqs = np.where(freqs >= 0)
    return freqs, power
    
def lowpass_filter(signal, freq_cutoff):
    """Return filtered signal with all frequencies above a cutoff frequency removed."""
    ft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))
    ft_filtered = np.where(np.abs(freqs)<freq_cutoff, ft, 0)
    signal_filtered = np.real(np.fft.ifft(ft_filtered))
    return signal_filtered

def fit_illumination_and_fringes(img, traces, yoffset, polydegree=7, aperture=25, sigma=3., freq_cutoff=0.025):
    """Return illumination model of a dewarped image."""
    x = np.arange(img.shape[0])
    y0_traces = np.array([trace.y[0] for trace in traces.traces])
    y0_traces_dewarped = y0_traces - yoffset
    
    illumination_model = np.full(img.shape, np.nan)
    fringes_model = np.full(img.shape, np.nan)
    
    for y0 in y0_traces_dewarped:
        aperture_window = np.arange(int(y0)-int(aperture/2.),
                                    int(y0)+int(aperture/2.)) - 1
        for i in aperture_window:
            row = img[i, :]
            is_valid = ~np.isnan(row)
            bestfit, outliers = robust_polyfit(x[is_valid],
                                             row[is_valid],
                                             deg=polydegree,
                                             return_outliers=True, sigma=sigma)
            illumination_model[i,:][is_valid] = bestfit

            residual = row[is_valid] - bestfit
            residual[outliers] = 0.
            freqs, power = get_power_spectrum(residual)
            fringes_model[i,:][is_valid] = lowpass_filter(residual, freq_cutoff)
    
    return illumination_model, fringes_model

def dewarp_all_flats(flats_fname, flats, flats_traces, dirout, new_shape=(1050,1050)):
    """Dewarp all flats."""
    if not os.path.exists(dirout):
        os.mkdir(dirout)
    
    ntotal = len(flats)
    for n, (fname, flat, traces) in enumerate(zip(flats_fname, flats, flats_traces), 1):
        print("Dewarping flat {0}/{1}: {2}.fits".format(n, ntotal, fname))
        transformer = EchelleImageTransformer(traces, new_shape)
        flat_dewarped = transformer.dewarp(flat)

        pathout = os.path.join(dirout, 'dewarped_'+fname+'.fits')
        print("Dewarping succeful! Saving dewarped flat as {0}\n".format(pathout))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.writeto(pathout, flat_dewarped,
                         output_verify="ignore", overwrite=True)
        
        matplotlib.cm.Greys_r.set_bad(color='black')
        figpathout = os.path.join(dirout, 'dewarped_'+fname+'.png')
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        plot_image(flat_dewarped, ax=ax, vmin=0, vmax=1e4)
        plt.tight_layout()
        plt.savefig(figpathout, dpi=250)
        plt.close()
        
def fit_illumination_and_fringes_all_flats(flats_fname, flats_dewarped, flats_traces,
                                           dirout_illumination, dirout_fringes,
                                           polydegree=7, aperture=25, sigma=3., freq_cutoff=0.025):
    """Make models."""
    if not os.path.exists(dirout_illumination):
        os.makedirs(dirout_illumination)
    
    if not os.path.exists(dirout_fringes):
        os.makedirs(dirout_fringes)
    
    nflats = len(flats_dewarped)
    for n, (fname, flat_dewarped, traces) in enumerate(zip(flats_fname, flats_dewarped, flats_traces), 1):
        print("Fitting illumination/fringes models {0}/{1}: {2}.fits".format(n, nflats, fname))
        transformer = EchelleImageTransformer(traces, newshape=flat_dewarped.shape)
        illumination_model_dewarped, fringes_model_dewarped = \
        fit_illumination_and_fringes(flat_dewarped, traces, transformer.yoffset)

        pathout = os.path.join(dirout_illumination, 'dewarped_illumination_model_'+fname+'.fits')
        print("Fit succeful! Saving dewarped illumination model as {0}".format(pathout))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.writeto(pathout, illumination_model_dewarped,
                         output_verify="ignore", overwrite=True)

        matplotlib.cm.Greys_r.set_bad(color='black')
        figpathout = os.path.join(dirout_illumination, 'dewarped_illumination_model_'+fname+'.png')
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        plot_image(illumination_model_dewarped, ax=ax, vmin=0, vmax=1e4)
        plt.tight_layout()
        plt.savefig(figpathout, dpi=250)
        plt.close()

        pathout = os.path.join(dirout_fringes, 'dewarped_fringes_model_'+fname+'.fits')
        print("Fit succeful! Saving dewarped fringes model as {0}\n".format(pathout))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.writeto(pathout, fringes_model_dewarped,
                         output_verify="ignore", overwrite=True)

        matplotlib.cm.Greys_r.set_bad(color='grey')
        figpathout = os.path.join(dirout_fringes, 'dewarped_fringes_model_'+fname+'.png')
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        plot_image(fringes_model_dewarped, ax=ax, vmin=-1e3, vmax=1e3)
        plt.tight_layout()
        plt.savefig(figpathout, dpi=250)
        plt.close()
        
def warp_illumination_and_fringes_all_flats(flats_fname, flats_traces,
                                            illumination_models_dewarped, fringes_models_dewarped,
                                            dirout_illumination, dirout_fringes):
    """Warp all."""
    if not os.path.exists(dirout_illumination):
        os.makedirs(dirout_illumination)
    
    if not os.path.exists(dirout_fringes):
        os.makedirs(dirout_fringes)
    
    nflats = len(flats_fname)
    for n, (fname, illumination_model_dewarped, fringes_model_dewarped, traces) in \
    enumerate(zip(flats_fname, illumination_models_dewarped, fringes_models_dewarped, flats_traces), 1):
        
        print("Warping illumination/fringes model {0}/{1}: {2}.fits".format(n, nflats, fname))
        transformer = EchelleImageTransformer(traces, newshape=illumination_model_dewarped.shape) 
        illumination_model = transformer.warp(illumination_model_dewarped)
        fringes_model = transformer.warp(fringes_model_dewarped)

        pathout = os.path.join(dirout_illumination, 'illumination_model_'+fname+'.fits')
        print("Warping succeful!")
        print("Saving illumination model as {0}".format(pathout))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.writeto(pathout, illumination_model,
                         output_verify="ignore", overwrite=True)

        matplotlib.cm.Greys_r.set_bad(color='black')
        figpathout = os.path.join(dirout_illumination, 'illumination_model_'+fname+'.png')
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        plot_image(illumination_model, ax=ax, vmin=0, vmax=1e4)
        plt.tight_layout()
        plt.savefig(figpathout, dpi=250)
        plt.close()

        pathout = os.path.join(dirout_fringes, 'fringes_model_'+fname+'.fits')
        print("Saving fringes model as {0}\n".format(pathout))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.writeto(pathout, fringes_model,
                         output_verify="ignore", overwrite=True)

        matplotlib.cm.Greys_r.set_bad(color='grey')
        figpathout = os.path.join(dirout_fringes, 'fringes_model_'+fname+'.png')
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        plot_image(fringes_model, ax=ax, vmin=-1e3, vmax=1e3)
        plt.tight_layout()
        plt.savefig(figpathout, dpi=250)
        plt.close()
        
def plot_fringecorr(flat, illumination_model, fringes_model, fpath):
    """Plot result fringes correction."""
    fig = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.,1.,1.], wspace=0.)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])
    ax3 = plt.subplot(gs[0,2])

    ax1, cbar1 = plot_image(flat-illumination_model, ax=ax1, vmin=4e2, vmax=-4e2, cmap='bwr', return_cbar=True)
    ax2, cbar2 = plot_image(fringes_model, ax=ax2, vmin=4e2, vmax=-4e2, cmap='bwr', return_cbar=True)
    ax3 = plot_image(flat-illumination_model-fringes_model, ax=ax3, vmin=4e2, vmax=-4e2, cmap='bwr')

    ax2.set_ylabel('')
    ax3.set_ylabel('')
    cbar1.remove()
    cbar2.remove()

    ax1.set_title('Flat - illumination', size=15)
    ax2.set_title('Fringes', size=15)
    ax3.set_title('Flat - illumination - fringes', size=15)

    plt.tight_layout()
    plt.savefig(fpath+'.png', dpi=250)
    plt.close()

def correct_fringes(flats_fnames, flats, illumination_models, fringes_models, dirout):
    """Blah."""
    if not os.path.exists(dirout):
        os.mkdir(dirout)

    for fname, flat, illumination_model, fringes_model in zip(flats_fnames,
                                                              flats,
                                                              illumination_models,
                                                              fringes_models):
        # do fringes correction
        fringes_model[np.isnan(fringes_model)] = 0.
        flat_fringescorr = flat - fringes_model
        
        # save corrected flats as fits files
        pathout = os.path.join(dirout, fname+'.fits')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.writeto(pathout, flat_fringescorr,
                         output_verify="ignore", overwrite=True)
        
        # plot result
        fpath = os.path.join(dirout, 'fringescorr' + fname)
        plot_fringecorr(flat, illumination_model, fringes_model, fpath)
        
def make_simple_badpixelmap(img, vmin, vmax):
    return np.array(np.logical_or(img < vmin, img > vmax), dtype=int)

def replace_badpixels(img, replacement, badpixelmap):
    """Replace badpixels."""
    img_corr = np.copy(img)
    badpixels = np.where(badpixelmap == 1)
    img_corr[badpixels] = replacement[badpixels]
    return img_corr