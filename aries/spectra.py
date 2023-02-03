import time
import signal
from .cleanspec import SpectralCube
from .preprocessing import plot_image
from .constants import TABLEAU20
from .constants import ARIES_NX, ARIES_NORDERS
from .constants import ARIES_GAIN, ARIES_RON
import Marsh
import GLOBALutils
from multiprocessing import Pool
import os
import scipy
import numpy as np
import warnings
import sys
from astropy.io import fits
from astropy import wcs

import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib import gridspec

sys.path.append('..')
base = os.path.abspath("../lib/ceres")+'/'
sys.path.append(base+"utils/Correlation")
sys.path.append(base+"utils/GLOBALutils")
sys.path.append(base+"utils/OptExtract")


class Timeout():
    """Timeout class using ALARM signal"""
    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()


TIMEOUT_LIMIT = 30  # s


class SpectralOrders:
    def __init__(self, data, **kwargs):
        self.data = data
        self.std = kwargs.pop('std', None)
        self.unit = kwargs.pop('unit', None)
        self.xunit = kwargs.pop('xunit', None)
        self.fname = kwargs.pop('fname', None)

        self.norders, self.nx = data.shape
        self.orders = np.arange(1, self.norders+1)
        self.x = kwargs.pop('x', np.arange(self.nx))

    def order(self, n, return_std=False):
        """Get spectral data of order n."""
        if n not in self.orders:
            raise ValueError('Cannot access invalid order n={}.'.format(n))

        if return_std:
            return self.std[n-1, :]
        else:
            return self.data[n-1, :]

    def plot(self, ax=None, yoffset=1, xmin=0, xmax=1, ymin=0, ymax=None):
        """Plot the extracted spectral orders."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        yticks = [n*yoffset for n in range(self.norders)]
        yticklabels = np.arange(1, self.norders+1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        for n in range(self.norders-1, 0, -1):
            ax.plot(yticks[n] + self.data[n, :] -
                    self.data[n, 0], color=TABLEAU20[n % 20])

        if ymax is None:
            ymax = np.max([yticks[n] + np.mean(self.data[n, :]) +
                          3*np.std(self.data[n, :]) for n in range(self.norders)])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(np.linspace(1, self.nx, 7))

        ax.set_xlabel('X (pixel)', size=15)
        ax.set_ylabel('Spectral order', size=15)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        return ax

    def plot_as_image(self, ax=None, **kwargs):
        """Return plot of spectral orders as image."""

        ax, cbar = plot_image(self.data, ax=ax, return_cbar=True,
                              extent=[0, 1, 0, 1], aspect=1,
                              **kwargs)

        ax.set_ylabel('Spectral order')
        ytick_step = 1./self.norders
        yticks = np.arange(ytick_step/2., 1, ytick_step)
        ytick_labels = np.arange(self.norders, 0, -1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        cbar.set_label('Flux (arbitrary units)', size=15)

        return ax, cbar


def plot_spectral_orders(data, ax=None, yoffset=1, xmin=0, xmax=1, ymin=0, ymax=None):
    """Plot the extracted spectral orders."""
    norders, nx = data.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    yticks = [n*yoffset for n in range(norders)]
    yticklabels = np.arange(1, norders+1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    for n in range(norders-1, 0, -1):
        ax.plot(yticks[n] + data[n, :] - data[n, 0], color=TABLEAU20[n % 20])

    if ymax is None:
        ymax = np.max([yticks[n] + np.nanmean(data[n, :]) + 3 *
                      np.nanstd(data[n, :]) for n in range(norders)])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.linspace(1, nx, 7))

    ax.set_xlabel('X (pixel)', size=15)
    ax.set_ylabel('Spectral order', size=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax


def plot_spectral_orders_as_image(data, ax=None, **kwargs):
    """Return plot of spectral orders as image."""
    norders, nx = data.shape
    ax, cbar = plot_image(data, ax=ax, return_cbar=True,
                          extent=[0, 1, 0, 1], aspect=1,
                          **kwargs)

    ax.set_ylabel('Spectral order')
    ytick_step = 1./norders
    yticks = np.arange(ytick_step/2., 1, ytick_step)
    ytick_labels = np.arange(norders, 0, -1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    cbar.set_label('Flux (arbitrary units)', size=15)

    return ax, cbar


def getSimpleSpectrum2(pars):
    trace_coeffs = pars[0]
    Aperture = pars[1]
    min_col = pars[2]
    max_col = pars[3]
    Result = Marsh.SimpleExtraction((GDATA.flatten()).astype('double'),
                                    scipy.polyval(trace_coeffs, np.arange(
                                        GDATA.shape[1])).astype('double'),
                                    GDATA.shape[0], GDATA.shape[1],
                                    GDATA.shape[1], Aperture, min_col, max_col)
    # After the function, we convert our list to a Numpy array.
    FinalMatrix = np.asarray(Result)
    return FinalMatrix


def simple_extraction(data, coefs, ext_aperture, min_extract_col, max_extract_col, npools):
    global GDATA
    GDATA = data
    npars_paralel = []
    for i in range(len(coefs)):
        npars_paralel.append([coefs[i, :], ext_aperture, int(
            min_extract_col[i]), int(max_extract_col[i])])

    p = Pool(npools)
    spec = np.array((p.map(getSimpleSpectrum2, npars_paralel)))
    p.terminate()

    #
    #         spec = []
    #         N = len(npars_paralel)+1
    #         for n, npars in enumerate(npars_paralel):
    #             result = getSimpleSpectrum2(npars)
    #             spec.append(result)
    return spec


def make_config_file_spectralextraction(fpath, img, trace, padding=0):
    """Make a spectral extraction configuration file specifying the x-range."""
    npixels = img.shape[0]
    xrange = np.arange(1, npixels+1)
    xmin_all, xmax_all = [], []

    coefs_all = trace['coefs_all']
    for coefs in coefs_all:
        center = scipy.polyval(coefs, xrange)
        is_valid = (0 <= center) * (center <= npixels)
        xmin_all.append(xrange[is_valid][0]+padding)
        xmax_all.append(xrange[is_valid][-1]-padding)

    orders = np.arange(len(coefs_all)) + 1
    header = 'xrange used for the spectral extraction for all orders \n' \
             'determined directly from the image data' \
             '1 order number 2 xmin 3 xmax'
    data = np.array([orders, xmin_all, xmax_all]).T
    np.savetxt(fpath, data, fmt='%i', header=header)


def PCoeff2(pars):
    trace_coeffs = pars[0]
    Aperture = pars[1]
    RON = pars[2]
    Gain = pars[3]
    NSigma = pars[4]
    S = pars[5]
    N = pars[6]
    Marsh_alg = pars[7]
    min_col = pars[8]
    max_col = pars[9]
    Result = Marsh.ObtainP((GDATA.flatten()).astype('double'),
                           scipy.polyval(trace_coeffs, np.arange(
                               GDATA.shape[1])).astype('double'),
                           GDATA.shape[0], GDATA.shape[1], GDATA.shape[1], Aperture, RON, Gain,
                           NSigma, S, N, Marsh_alg, min_col, max_col)

    # After the function, we convert our list to a Numpy array.
    FinalMatrix = np.asarray(Result)
    # And return the array in matrix-form.
    FinalMatrix.resize(GDATA.shape[0], GDATA.shape[1])
    return FinalMatrix


def obtain_P(data, trace_coeffs, Aperture, RON, Gain, NSigma, S, N, Marsh_alg, min_col, max_col, npools):
    global GDATA
    GDATA = data
    npars_paralel = []
    for i in range(len(trace_coeffs)):
        npars_paralel.append([trace_coeffs[i, :], Aperture, RON, Gain,
                             NSigma, S, N, Marsh_alg, int(min_col[i]), int(max_col[i])])

    p = Pool(npools)
    spec = np.array((p.map(PCoeff2, npars_paralel)))
    p.terminate()
    return np.sum(spec, axis=0)


def getSpectrum2(pars):
    trace_coeffs = pars[0]
    Aperture = pars[1]
    RON = pars[2]
    Gain = pars[3]
    S = pars[4]
    NCosmic = pars[5]
    min_col = pars[6]
    max_col = pars[7]
    Result, size = Marsh.ObtainSpectrum((GDATA.flatten()).astype('double'),
                                        scipy.polyval(trace_coeffs, np.arange(
                                            GDATA.shape[1])).astype('double'),
                                        P.flatten().astype(
                                            'double'), GDATA.shape[0],
                                        GDATA.shape[1], GDATA.shape[1], Aperture, RON,
                                        Gain, S, NCosmic, min_col, max_col)
    # After the function, we convert our list to a Numpy array.
    FinalMatrix = np.asarray(Result)
    # And return the array in matrix-form.
    FinalMatrix.resize(3, size)
    return FinalMatrix


def optimal_extraction(data, Pin, coefs, ext_aperture, RON, GAIN, MARSH, COSMIC, min_extract_col, max_extract_col, npools):
    global GDATA, P
    P = Pin
    GDATA = data
    npars_paralel = []
    for i in range(len(coefs)):
        npars_paralel.append([coefs[i, :], ext_aperture, RON, GAIN, MARSH, COSMIC, int(
            min_extract_col[i]), int(max_extract_col[i])])

    p = Pool(npools)
    spec = np.array((p.map(getSpectrum2, npars_paralel)))
    p.terminate()
    return spec


def optext_all(fnames, imgs, imgs_traces,
               dirout, npools, optext_params, replace_bad_weights=False):
    if not os.path.exists(dirout):
        os.mkdir(dirout)

    ntotal = len(imgs)
    for n, (fname, img, trace) in enumerate(zip(fnames, imgs, imgs_traces), 1):

        # filter out any remaining NaNs
        img[np.isnan(img)] = 0.

        print("Optimal extraction {0}/{1}: {2}.fits".format(n, ntotal, fname))
        print("\tCalculating weights...")
        # Obtain P, weights used for the optimal extraction
        # Run block of code with timeouts
        solved_weights = False
        NITER_MAX = 10
        niter = 0
        while not solved_weights:
            try:
                with Timeout(TIMEOUT_LIMIT):
                    weights_optext = obtain_P(data=img,trace_coeffs=trace.coefs_all,Aperture=optext_params['aperture'],RON=ARIES_RON,Gain=ARIES_GAIN,NSigma=optext_params['nsigma'],
                    S=optext_params['interp_fraction'],
                    N=optext_params['polydegree'],
                    Marsh_alg=optext_params['use'],
                    min_col=optext_params['min_col'],
                    max_col=optext_params['max_col'],
                    npools=npools
                    )
                    solved_weights = True
            except Timeout.Timeout:
                if niter == NITER_MAX:
                    raise ValueError('Timed out.')

                optext_params['min_col'] += 10
                optext_params['max_col'] -= 10
                niter += 1
                print('Timed out. Extending range by +-10 pixels. Trying again.')

        if replace_bad_weights:
            weights_optext[np.isnan(weights_optext)] = 1. / \
                optext_params['aperture']  # uniform weights

        pathout = os.path.join(dirout, 'weights_optext_'+fname+'.fits')
        print(
            "\tDone. Saving optimal extraction weights as {0}".format(pathout))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.writeto(pathout, weights_optext,
                         output_verify="ignore", overwrite=True)

        # Plot weights
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax = plot_image(weights_optext, ax=ax, cmap='magma',
                        vmin=0., vmax=1./optext_params['aperture'])
        ax.set_title('Weights optimal extraction', size=15)
        fpath = os.path.join(dirout, 'plot_weights_optext_'+fname)
        plt.tight_layout()
        plt.savefig(fpath+'.png', dpi=200)
        plt.close()

        # Perform optimal extraction
        print("\tExtracting spectra...")
        result_optext = optimal_extraction(
            data=img,
            Pin=weights_optext,
            coefs=trace.coefs_all,
            ext_aperture=optext_params['aperture'],
            RON=ARIES_RON,
            GAIN=ARIES_GAIN,
            MARSH=optext_params['interp_fraction'],
            COSMIC=optext_params['ncosmic'],
            min_extract_col=optext_params['min_col'],
            max_extract_col=optext_params['max_col'],
            npools=npools
        )
        pathout = os.path.join(dirout, 'result_optext_'+fname+'.fits')
        print(
            '\tDone. Saving optimal extraction result as {0}\n'.format(pathout))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.writeto(pathout, result_optext,
                         output_verify="ignore", overwrite=True)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax = plot_spectral_orders(result_optext[:, 1], ax=ax, yoffset=1.)
        fpath = os.path.join(dirout, 'plot_spectral_orders_'+fname)
        plt.suptitle(fname+'.fits (optimal extraction)', size=15)
        plt.tight_layout()
        plt.savefig(fpath+'.png', dpi=200)
        plt.close()

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax = plot_spectral_orders_as_image(result_optext[:, 1], ax=ax, vmin=0, vmax=np.quantile(
            result_optext[:, 1], 0.997), cmap='plasma')
        fpath = os.path.join(dirout, 'plot_spectral_orders_as_image_'+fname)
        plt.suptitle(fname+'.fits (optimal extraction)', size=15)
        plt.tight_layout()
        plt.savefig(fpath, dpi=200)
        plt.close()


def make_spectral_cube_aries(fpaths):
    """Make a SpectralCube from the output files of optimal extraction."""
    nobs = len(fpaths)

    # check output shape
    result_optext = fits.getdata(fpaths[0])
    norders, _, nx = result_optext.shape
    if norders != ARIES_NORDERS:
        warnings.warn('Number of orders = {} != {}'.format(
            norders, ARIES_NORDERS))
    if nx != ARIES_NX:
        warnings.warn('Nx = {} != {}'.format(nx, ARIES_NX))

    flux = np.zeros(shape=(norders, nobs, nx))
    var = np.zeros(shape=(norders, nobs, nx))
    for i, f in enumerate(fpaths):
        result_optext = fits.getdata(f)
        flux[:, i, :] = result_optext[::-1, 1, :]
        var[:, i, :] = result_optext[::-1, 2, :]

    # Remove orders which contain only zeros
    mask = np.ones(norders, dtype=bool)
    for norder in range(norders):
        if np.all(flux[norder, :, :] == 0.):
            mask[norder] = False
    flux = flux[mask, :, :]
    var = var[mask, :, :]

    return SpectralCube(flux, var=var)
