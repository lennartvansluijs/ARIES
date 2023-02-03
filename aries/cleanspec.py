import os
import pickle
import warnings

import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib import gridspec

import numpy as np
import numpy.ma as ma

from astropy.io import fits
from astropy import wcs
from astropy.constants import h, k_B, c
import astropy.units as u
from itertools import islice
from collections import deque
import scipy.signal as signal

from .preprocessing import make_point_footing, robust_polyfit
from scipy.ndimage import generic_filter, gaussian_laplace, median_filter
from scipy.optimize import curve_fit, OptimizeWarning
from .constants import TABLEAU20
from .crosscorrelate import apply_rv, regrid_spectrum

def simulate_planet_spectrum(template_wav, template_spec, data_wav, rvplanet, mode='1D'):
    model = np.zeros(shape=(len(rvplanet), len(data_wav)))
    if mode == '1D':
        for i, rv in enumerate(rvplanet):
            template_wav_shifted = apply_rv(template_wav, rv=rv) #  shift to the planet velocity
            _, model[i,:] = regrid_spectrum(template_wav_shifted, template_spec, data_wav)
        return model
    elif mode == '2D':
        for i, rv in enumerate(rvplanet):
            template_wav_shifted = apply_rv(template_wav, rv=rv) #  shift to the planet velocity
            _, model[i,:] = regrid_spectrum(template_wav_shifted, template_spec[i,:], data_wav)
        return model
    else:
        raise ValueError('Invalid mode.')
    

def simulate_planet_spectrum_2D(template_wav, template_spec, data_wav, rvplanet):
    model = np.zeros(shape=(len(rvplanet), len(data_wav)))
    for i, rv in enumerate(rvplanet):
        template_wav_shifted = apply_rv(template_wav, rv=rv) # shift to the planet velocity
        _, model[i,:] = regrid_spectrum(template_wav_shifted, template_spec[i,:], data_wav)
    return model


def planck(wav, T, return_flux=False, unit_system='si'):
    """Return Planck function for given wavelength range (in meter) and temperature (in Kelvin)."""
    wav *= u.meter
    T *= u.Kelvin
    a = 2.0*h*c**2
    b = (h*c)/(wav*k_B*T)
    intensity = (a/wav**5)/(np.exp(b) - 1.)
    if return_flux:
        return getattr(intensity * np.pi, unit_system).value # correction for integration over solid angle
    else:
        return getattr(intensity, unit_system).value

def correct_bads(data, badpixelmap, interp_direction='horizontal'):
    """Correct badpixels in spectral cube data."""
    ny, nx = data.shape
    data_corr = np.copy(data)
    if interp_direction == 'horizontal':
        x = np.arange(1, nx+1)
        for row in range(ny):
            y = data[row, :]
            bads = badpixelmap[row, :]
            data_corr[row, bads] = np.interp(x=x[bads], xp=x[~bads], fp=y[~bads]) # use good pixels to interpolate over bads
        return data_corr
    elif interp_direction == 'vertical':
        x = np.arange(1, ny+1)
        for col in range(nx):
            y = data[:, col]
            bads = badpixelmap[:, col]
            data_corr[bads, col] = np.interp(x=x[bads], xp=x[~bads], fp=y[~bads]) # use good pixels to interpolate over bads
        return data_corr
    else:
        raise ValueError('Invalid interpolation direction. Valid directions are "horizontal" or "vertical".')

def identify_badcolumns(data, sigma=5, medfilt_size=5):
    """Return list of indices of badcolumns in image."""
    row_median = np.median(data, axis=0)
    smooth_model = median_filter(row_median, size=medfilt_size,
                                 mode='constant', cval=np.median(row_median))
    residual = row_median - smooth_model
    badcolumns = np.where(np.abs(residual) > sigma*np.std(residual))[0]
    return badcolumns

def correct_badcolumns(img, badcolumns):
    """Return image corrected for bad column."""
    img_corr = np.copy(img)
    xrange = np.arange(img.shape[1])
    goodcols = np.setxor1d(xrange, badcolumns)
    for column in badcolumns:
        closest_neighbouring_columns = np.argsort(np.abs(goodcols-column))[1:3]
        img_corr[:, column] = (img[:, closest_neighbouring_columns[0]] + \
                               img[:, closest_neighbouring_columns[1]])/2.
    return img_corr


def crosscorrelate(x, y, xt, yt, xs, left=0, right=0):
    cc = np.zeros(len(xs))
    for i, s in enumerate(xs):
        yi = np.interp(x=x, xp=xt-s, fp=yt, left=left, right=right)
        cc[i] = np.sum(y * yi)
    return cc

# def calculate_ccmatrix(data, x, xt, yt, xs):
#     nx, ny = data.shape
#     ccmatrix = np.zeros(shape=(nx,len(xs)))
#     for j in range(nx):
#         ccmatrix[j, :] = crosscorrelate(x, data[j, :], xt, yt, xs)
#     return ccmatrix

# def calculate_ccmatrix(data, x, xt, yt, xs, mode ='default'):
#     if mode == 'default':
#         nx, ny = data.shape
#         ccmatrix = np.zeros(shape=(nx,len(xs)))
#         for j in range(nx):
#             ccmatrix[j, :] = crosscorrelate(x, data[j, :], xt, yt, xs)
#     elif mode == 'numpy':
#         # define some relevant parameters
#         dx = int(xs.max())
#         nshifts = dx*2+1
#         nx, ny = data.shape
#         ccmatrix = np.zeros(shape=(nx,nshifts))

#         # trim edges of template to allow only valid shifts (where both spectra fully overlap)
#         yt = yt[dx:-dx]
#         for j in range(nx):
#             ccmatrix[j,:] = np.correlate(data[j,:], yt, 'valid')[::-1]
#     else:
#         raise ValueError('Invalid mode: {}.'.format(mode))
#     return ccmatrix

def clip_mask(data, mask, return_clip=False):
    """Return clipped data."""
    clip = np.any(mask, axis=0)
    data_c = data[:, ~clip]
    if return_clip:
        return data_c, clip
    else:
        return data_c

# def calculate_ccmatrix(data, mask, dx):
#     """Return cross-correlation matrix."""
#     # clip masked data
#     data_clipped = clip_mask(data, mask)

#     # define some relevant parameters
#     nshifts = dx*2+1
#     nobs = data.shape[0]
#     ccmatrix = np.zeros(shape=(nobs,nshifts))

#     # define and trim edges of template to allow only 
#     # valid shifts (where both spectra fully overlap)
#     TEMPLATE_INDEX = 0 #maybe mean or highest snr?
#     TRIMMED_EDGES = slice(dx,-dx,1)
#     template = data_clipped[TEMPLATE_INDEX,TRIMMED_EDGES]
    
#     # cross-correlate
#     for j in range(nobs):
#         ccmatrix[j,:] = np.correlate(data_clipped[j,:], template, 'valid')[::-1] # increase to 20 or 30
#     return ccmatrix

def gaussian(x, *p):
    amp, mu, sigma, y0 = p
    return y0+amp*np.exp(-(x-mu)**2/(2.*sigma**2))


class SpectralCube:
    def __init__(self, data, **kwargs):
        self.data = data
        self.norders, self.nobs, self.npixels = data.shape        
        self.target = kwargs.pop('target', None)
        self.var = kwargs.pop('var', None)
        self.mask = kwargs.pop('mask', None)
    
    @property
    def data_ma(self):
        return ma.masked_array(self.data, self.mask)
    
    def apply_mask(self, apply_to_value):
        """Apply a mask to any column containing a specified value."""
        self.mask = np.array(np.broadcast_to(
            np.any(np.array(self.data == apply_to_value), axis=1)[:, np.newaxis, :],
            self.data.shape), dtype=int)
        
    def get_spectralorder(self, norder):
        return SpectralOrder(norder=norder, data=self.data[norder-1,:,:], mask=self.mask[norder-1,:,:], target=self.target)
        
    @classmethod
    def load(cls, fpath):
        """Load SpectralCube object."""
        # Load data
        data = fits.getdata(fpath)
        
        # Load kwargs from header
        hdr = fits.getheader(fpath)
        kwargs = {}
        if 'TARGET' in hdr.keys():
            kwargs['target'] = hdr['TARGET']

        # Load variance/mask data, if file exists
        items = [('var', '_var'), ('mask', '_mask')]
        for (item, extension) in items:
            fpath_item = os.path.splitext(fpath)[0] + extension + os.path.splitext(fpath)[1]
            if os.path.exists(fpath_item):
                kwargs[item] = fits.getdata(fpath_item)
            else:
                kwargs[item] = None
        
        return cls(data, **kwargs)
    
    def save(self, fpath):
        """Save SpectralCube object."""
        # Prepare fits header
        w = wcs.WCS(naxis=3)
        w.wcs.crpix = [0, 0, 0]
        w.wcs.cdelt = np.array([1, 1, 1])
        w.wcs.ctype = ["PIXEL", "NOBS", "SPECORDER"]
        hdr = w.to_header()
        hdr.set('target', self.target, 'target name')
        hdu = fits.PrimaryHDU(header=hdr)
        
        # Save data as .fits file
        fits.writeto(fpath, self.data, header=hdr, overwrite=True)
        
        # Save variance/mask as .fits file, if variance/mask is defined
        items = [(self.var, '_var'), (self.mask, '_mask')]
        for (item, extension) in items:
            if item is not None:
                fpath_item = os.path.splitext(fpath)[0] + extension + os.path.splitext(fpath)[1]
                try:
                    fits.writeto(fpath_item, item, header=hdr, overwrite=True)
                except KeyError:
                    fits.writeto(fpath_item, np.array(item, dtype=int))
            else:
                pass
    
    @property
    def std(self):
        """Return standard deviation of data."""
        if self.var is None:
            raise ValueError('Variance not defined.')
        else:
            return np.sqrt(self.var)
        
    def plot(self, data='default', apply_mask=True, mask_color='grey', vmin=None, vmax=None, cmap='hot', return_cbar=False, origin='upper', figtitle='default', **kwargs):
        """Plot overview of spectral cube."""
        plt.figure(figsize=(10,10))
        if data is 'default':
            data = self.data
        if apply_mask:
            data = ma.masked_array(data, self.mask)
            cmap = plt.get_cmap(cmap)
            cmap.set_bad(mask_color,1.)
        if figtitle is 'default':
            figtitle = 'Spectral Time Series All Orders'
        if self.target is not None:
            figtitle += ' ({})'.format(self.target)
        if vmin is None:
            vmin = np.mean(data)-3*np.std(data)
        if vmax is None:
            vmax = np.mean(data)+3*np.std(data)
        
        height_ratios = np.ones(self.norders+2)
        height_ratios[[-1, -2]] = [0.5, 3]
        hspace = 0.25
        
        gs = gridspec.GridSpec(self.norders+2, 1, height_ratios=height_ratios, hspace=hspace)
        axes = [plt.subplot(gs[n, 0]) for n in range(self.norders+2)]
        
        for n in range(self.norders):
            aspect = self.npixels/float(self.nobs*(self.norders+2))/(1.+hspace)
            im = axes[n].imshow(data[n,:,:], vmin=vmin, vmax=vmax, cmap=cmap,
                           extent=[1,self.npixels,1,self.nobs], aspect=aspect, origin=origin, **kwargs)
            axes[n].set_yticks([])
            axes[n].set_xticks([])
            axes[n].annotate(n+1, xy=(-0.01, 0.5),  xycoords='axes fraction',
                            xytext=(-0.01, 0.5), va='center', ha='right', textcoords='axes fraction', size=12)
            if n == 0:
                axes[n].set_title(figtitle, fontsize=18, pad=15)
                axes[n].annotate('', xy=(1.0125, 1),  xycoords='axes fraction',
                            xytext=(1.0125, 0), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', width=0.15, headwidth=5, headlength=5))
                axes[n].annotate('Time', xy=(1.025, 0.5),  xycoords='axes fraction',
                            xytext=(1.025, 0.5), va='center', textcoords='axes fraction', size=15)#, rotation=90)
            if n == self.norders-1:
                axes[n].set_xticks(np.linspace(0, self.npixels, 9))
                axes[n].tick_params(labelsize=12)
                axes[n].set_xlabel('X (pixel)', size=15)
            if n == int( (self.norders-1)/2):
                axes[n].set_ylabel('Spectral Order', size = 15, labelpad=25)
        
        axes[-2].axis('off')
        cbar = Colorbar(ax = axes[-1], mappable = im, orientation = 'horizontal', ticklocation = 'bottom', extend='both')
        cbar.set_label(r'Flux (arbitrary units)', fontsize=15)
        cbar.ax.tick_params(labelsize=12)
        
        if return_cbar:
            return (axes, cbar)
        else:
            return axes
        
    def normalized_rows(self):
        if self.mask is None:
            return np.median(self.data, axis=2)[:,:,np.newaxis]
        else:
            return ma.median(self.data_ma, axis=2)[:,:,np.newaxis].data
            
    def mean_columns(self):
        if self.mask is None:
            return np.mean(self.data, axis=1)[:,np.newaxis,:]
        else:
            return ma.mean(self.data_ma, axis=1)[:,np.newaxis,:].data
    
    def gaussian_laplace(self, sigma):
        data_gl = np.zeros(shape=self.data.shape)
        for n in range(self.norders):
            data_gl[n,:,:] = gaussian_laplace(self.data[n,:,:], sigma=sigma)
        return data_gl

    def plot_spectral_order(self, norder, data='default', apply_mask=True,
                            mask_color='grey', vmin=None, vmax=None, cmap='hot',
                            return_cbar=False, origin='upper',
                            figtitle='default', **kwargs):
        """Plot overview of spectral order."""
        if norder not in np.arange(1, self.norders+1):
            raise IndexError('Spectral order number {} outside range.'.format(norder))
        if data is 'default':
            data = self.data
        if figtitle is 'default':
            figtitle = 'Spectral Time Series'
        figtitle += ' (order = {})'.format(norder)
        if self.target is not None:
            figtitle += ' ({})'.format(self.target)
        if apply_mask:
            data = ma.masked_array(data, self.mask)
            cmap = plt.get_cmap(cmap)
            cmap.set_bad(mask_color,1.)
        if vmin is None:
            vmin = np.mean(data)-3*np.std(data)
        if vmax is None:
            vmax = np.mean(data)+3*np.std(data)

        sf = 1.68
        fig = plt.figure(figsize=(5.5*sf, 5.5))
        height_ratios = [1, 0.05]
        hspace = sf/5
        aspect = self.npixels/float(self.nobs)/(sf+hspace)

        gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=hspace)
        axes = [plt.subplot(gs[n, 0]) for n in range(2)]

        ax = axes[0]
        im = ax.imshow(data[norder-1,:,:], vmin=vmin, vmax=vmax, cmap=cmap,
                       extent=[1,self.npixels,1,self.nobs], aspect=aspect, origin=origin, **kwargs)

        ax.set_xticks(np.linspace(0, self.npixels, 9))
        ax.tick_params(labelsize=12)
        ax.set_xlabel('X (pixel)', size=15)
        ax.set_ylabel('Frame number', size = 15, labelpad=15)
        ax.set_title(figtitle, fontsize=18, pad=10)

        # Create colorbar
        cbar = Colorbar(ax = axes[-1], mappable = im, orientation = 'horizontal', ticklocation = 'bottom', extend='both')
        cbar.set_label(r'Flux (arbitrary units)', fontsize=15)
        cbar.ax.tick_params(labelsize=12)
        
        if return_cbar:
            return (axes, cbar)
        else:
            return axes
        
    def plot_all_orders(self, *args, **kwargs):
        for norder in range(1,self.norders+1):
            ax = self.plot_spectral_order(norder, *args, **kwargs)
            plt.show()
    
    def detect_blobs(self, blob_scale=0.5, filter_size=11, sigma=5, silent=False, return_full=False):
        """Return boolean np.array map of pixels identified as blobs by blob detection
        algorithm an Gaussian laplace of data."""
        gl = self.gaussian_laplace(sigma=blob_scale)

        blobmap = np.zeros(shape=self.data.shape, dtype=bool)
        sigmamap = np.zeros(shape=self.data.shape)
        for n in range(self.norders):
            if not silent:
                print('Blob detection: order {}/{}'.format(n+1, self.norders))
            data_gl = gl[n,:,:]

            point_footing = make_point_footing(filter_size)
            local_median = median_filter(data_gl, footprint=point_footing)
            local_std = generic_filter(data_gl-local_median, footprint=point_footing, function=np.nanstd)

            local_std_ma = ma.masked_array(local_std, local_std==0) # Avoid divide by zero
            z = np.abs(data_gl-local_median)/local_std_ma
            is_blob = (z > sigma)

            sigmamap[n,:,:] = z.data
            blobmap[n,:,:] = np.array(is_blob, dtype=bool)
        
        if return_full:
            return blobmap, gl, sigmamap
        else:
            return blobmap
    
    def correct_badpixels(self, badpixelmap, interp_direction='horizontal'):
        """Correct badpixels in data by column/row wise interpolation."""
        for norder in range(1, self.norders+1):
            self.data[norder-1,:,:] = correct_bads(self.data[norder-1,:,:],
                                                   badpixelmap[norder-1,:,:],
                                                   interp_direction)
            
    def correct_badcolumns(self, sigma=5., medfilt_size=3):
        """Detect and correct badcolumns in the data."""
        data_corr = np.zeros(shape=self.data.shape)
        for norder in range(1, self.norders+1):
            badcolumns = identify_badcolumns(self.data[norder-1,:,:], sigma, medfilt_size)
            data_corr[norder-1,:,:] = correct_badcolumns(self.data[norder-1,:,:], badcolumns)
        self.data = data_corr
        
    def align_all(self, dxmax, osr=10, plot=True, silent=True, ccmode='numpy', dirout=None):
        """Align all spectra."""
        # allocate memory for realigned data
        data_aligned = np.zeros(self.data.shape)
        xshift = np.linspace(-dxmax, dxmax, dxmax*2+1)
        
        # x-axis data
        x = np.arange(self.npixels)
        drifts = np.zeros(shape=(self.norders, self.nobs))
        for norder in range(1,self.norders+1):
            
            if not silent:
                print('Aligining spectral order {}/{}'.format(norder, self.norders))
            
            # use first spectrum as template
            template = self.data[norder-1,0,:]
            
            # cross correlate 
            #             ccmatrix = calculate_ccmatrix(data=self.data[norder-1,:,:],
            #                                           x=x, xt=x, yt=template, xs=xshift, mode=ccmode)
            ccmatrix = calculate_ccmatrix(data=self.data[norder-1,:,:],
                                          mask=self.mask[norder-1,:,:], dx=dxmax)
            ccmatrix_n = np.abs(ccmatrix / np.sum(ccmatrix, axis=1)[:,np.newaxis]) #maybe not required
            
            if plot:
                plt.figure(figsize=(16.8, 10))
                gx, gy = int(np.ceil(np.sqrt(self.nobs))), int(np.ceil(np.sqrt(self.nobs)))
                gs = gridspec.GridSpec(gx, gy)
                axes = []
                for n in range(gx):
                    for m in range(gy):
                        axes.append(plt.subplot(gs[n, m]))
                for n in range(len(axes)):
                    axes[n].set_yticks([])
                    #axes[n].set_xticks([])
                    if n+1 > self.nobs:
                        axes[n].axis('off')
            
            # fit for drift
            drift = np.zeros(self.nobs)
            for i in range(self.nobs):

                # initial guess
                y = ccmatrix_n[i,:]
                xos = np.linspace(xshift[0], xshift[-1], osr*len(xshift))
                amp0 = np.max(y)
                mu0 = xos[np.argmax(y)]
                sigma0 = 1.
                y0 = 0.
                p0 = [amp0, mu0, sigma0, y0]

                # best fit
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    try:
                        coefs, var_matrix = curve_fit(gaussian, xshift, y, p0=p0)
                    except (RuntimeError, OptimizeWarning) as e:
                        coefs = p0 # set to initial guess
                drift[i] = coefs[1]
                
                if plot:
                    # best fit
                    axes[i].scatter(xshift, y, color='k', s=5, label='CC')
                    axes[i].plot(xos, gaussian(xos, *coefs), color=TABLEAU20[6])
                    axes[i].axvline(drift[i], color=TABLEAU20[6], ls='--')
                    axes[i].set_ylim(y.min(),y.max())
            
            # outlier correction
            x = np.arange(len(drift))
            valid = (np.abs(drift) <= dxmax)
            drift = np.interp(x, x[valid], drift[valid])
            
            if plot:
                if dirout is not None:
                    # plot previous figure
                    fname = 'alignment_fit_order_{}'.format(norder)+'.png'
                    pathout = os.path.join(dirout, fname)
                    plt.savefig(pathout, dpi=250)
                    plt.close()
                else:
                    plt.show()
                
                # second figure
                plt.figure()
                
                # some plot parameters
                obsid = np.arange(1, self.nobs+1)
                aspect = (ccmatrix_n.shape[1]/ccmatrix_n.shape[0])/1.68

                ax = plt.imshow(ccmatrix_n, aspect=aspect,
                                extent=[xshift[0]-0.5, xshift[-1]+0.5,
                                        0.5, self.nobs+0.5], cmap='Blues_r')
                plt.plot(drift, obsid, color='k', lw=0.5, label='Drift')
                cbar = plt.colorbar()
                cbar.set_label('CC', size=15)
                plt.title('CC Matrix (order={})'.format(norder), size=15)
                plt.xlabel('Shift [pixel]', size=15)
                plt.ylabel('# observation', size=15)
                leg = plt.legend(frameon=False)
                
                if dirout is not None:
                    fname = 'cc_matrix_order_{}'.format(norder)
                    pathout = os.path.join(dirout, fname)
                    plt.savefig(pathout, dpi=250)
                    plt.close()
                else:
                    plt.show()
            
            # realign data
            data_aligned[norder-1,:,:] = realign(self.data[norder-1,:,:], drift)
            drifts[norder-1,:] = drift
            
        return data_aligned, drifts

def plot_spectral_order(data, norder='', target=None, mask=None, apply_mask=True,
                            mask_color='grey', vmin=None, vmax=None, cmap='hot',
                            return_cbar=False, origin='upper',
                            figtitle='default', **kwargs):
        """Plot overview of spectral order."""
        if figtitle is 'default':
            figtitle = 'Spectral Time Series'
        figtitle += ' (order = {})'.format(norder)
        if target is not None:
            figtitle += ' ({})'.format(target)
        if apply_mask:
            data = ma.masked_array(data, mask)
            cmap = plt.get_cmap(cmap)
            cmap.set_bad(mask_color,1.)
        if vmin is None:
            vmin = np.mean(data)-3*np.std(data)
        if vmax is None:
            vmax = np.mean(data)+3*np.std(data)

        sf = 1.68
        fig = plt.figure(figsize=(5.5*sf, 5.5))
        height_ratios = [1, 0.05]
        hspace = sf/5
        nobs, npixels = data.shape
        aspect = npixels/float(nobs)/(sf+hspace)

        gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=hspace)
        axes = [plt.subplot(gs[n, 0]) for n in range(2)]

        ax = axes[0]
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap,
                       extent=[1,npixels,1,nobs], aspect=aspect, origin=origin, **kwargs)

        ax.set_xticks(np.linspace(0, npixels, 9))
        ax.tick_params(labelsize=12)
        ax.set_xlabel('X (pixel)', size=15)
        ax.set_ylabel('Frame number', size = 15, labelpad=15)
        ax.set_title(figtitle, fontsize=18, pad=10)

        # Create colorbar
        cbar = Colorbar(ax = axes[-1], mappable = im, orientation = 'horizontal', ticklocation = 'bottom', extend='both')
        cbar.set_label(r'Flux (arbitrary units)', fontsize=15)
        cbar.ax.tick_params(labelsize=12)
        
        if return_cbar:
            return (axes, cbar)
        else:
            return axes

def get_mask(data, threshold, envelope_deg=3, mode='sigma'):
    """Return mask of noisy columns."""
    mask = np.zeros(shape=data.shape)
    std_cols = np.std(data, axis=0)
    ncols = len(std_cols)
    envelope = robust_polyfit(x=np.arange(ncols), y=std_cols, sigma=5., deg=envelope_deg)
    std_cols_corr = std_cols / envelope

    if mode is 'sigma':
        badcols = (std_cols_corr > threshold)
        mask[:,badcols] = 1.
    elif mode is 'ratio':
        badcols = np.argsort(std_cols_corr)[-int(threshold*ncols):]
        mask[:,badcols] = 1.
    else:
        raise ValueError('Invalid mask mode.')
    return mask

def calc_ccmatrix(data, template, shifts, mask):
    
    if mask is None:
        mask = np.zeros(data.shape)
    MASK = ~np.any(mask, axis=0)

    nshifts = len(shifts)
    nobs = int(data.shape[0])
    ccmatrix = np.zeros(shape=(nobs, nshifts))
    
    wav = np.arange(len(template))
    for i in range(nobs):
        spec = data[i,:]
        for j, dx in enumerate(shifts):
            _, template_s = regrid_spectrum(wav, template, wav+dx)
            ccmatrix[i, j] = np.corrcoef(template_s[MASK], spec[MASK])[0, 1] #Pearson
    return ccmatrix

def realign(data, drift):
    """Return realigned spectra."""
    data_aligned = np.copy(data)
    nobs, npixels = data.shape
    x = np.arange(0, npixels)
    for i in range(nobs):
        data_aligned[i,:] = np.interp(x=np.arange(0,npixels),
                                      xp=np.arange(0,npixels)+drift[i],
                                      fp=data[i,:])
    return data_aligned

class SpectralOrder:
    
    def __init__(self, data, norder, **kwargs):
        self.data = data
        self.norder = norder
        self.nobs, self.nx = data.shape
        self.target = kwargs.pop('target', None)
        self.mask = kwargs.pop('mask', np.zeros(data.shape, dtype=bool))
        self.wavsolution = kwargs.pop('wavsolution', (np.arange(1, self.nx+1), None))
        self.time = kwargs.pop('time', None)
        self.phase = kwargs.pop('phase', None)
        self.vbary = kwargs.pop('vbary', None)
        self.obsdate = kwargs.pop('obsdate', None)
        self.error = kwargs.pop('error', np.ones(data.shape)*np.nan) # individual errors
    
    def plot(self, data='default', apply_mask=True, mask=None, xunit='pixel',
             wavsolution=None, mask_color='grey', vmin=None, vmax=None, cmap='inferno',
             return_cbar=False, origin='upper', yunit='frame',
             figtitle='default', ax=None, add_cbar=False, **kwargs):
        """Plot overview of spectral order."""
        if data is 'default':
            data = self.data
        if figtitle is 'default':
            figtitle = 'Spectral Time Series (order = {})'.format(self.norder)
            if self.target is not None:
                figtitle += ' ({})'.format(self.target)
        if apply_mask:
            if mask is None:
                mask = self.mask
            data = ma.masked_array(data, mask)
            cmap = plt.get_cmap(cmap)
            cmap.set_bad(mask_color,1.)
        if vmin is None:
            vmin = np.quantile(data, 0.003) # Quantile(mu - 3 sigma)
        if vmax is None:
            vmax = np.quantile(data, 0.997) # Quantile(mu + 3 sigma)

        sf = 1.68*2
        fig = plt.figure(figsize=(5*sf, 5))
        height_ratios = [1, 0.05]
        hspace = sf/5
        nobs, npixels = data.shape
        aspect = npixels/float(nobs)/(sf+hspace)

        if ax is None:
            gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=hspace)
            axes = [plt.subplot(gs[n, 0]) for n in range(2)]
            ax = axes[0]
            add_cbar = True
        
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap,
                       extent=[0,npixels,0,nobs], aspect=aspect, origin=origin, **kwargs)

        if xunit is 'pixel':
            ax.set_xticks(np.linspace(0, self.nx, 9))
            ax.tick_params(labelsize=12)
            ax.set_xlabel('X (pixel)', size=15)
            ax.set_xticklabels([ int(pixel) for pixel in np.linspace(self.wavsolution[0][0], self.wavsolution[0][-1], 9)])
        if xunit is 'micron':
            if self.wavsolution is not None:
                ax.set_xticks(np.linspace(0, self.nx, 9))
                ax.set_xticklabels([ "{:.3f}".format(wav) for wav in np.linspace(self.wavsolution[1][0], self.wavsolution[1][-1], 9)])
                ax.tick_params(labelsize=12)
                ax.set_xlabel('Wavelength (micron)', size=15)
            else:
                raise ValueError('Wavsolution has not been defined.')
        
        if yunit is 'frame':
            ax.set_ylabel('Frame number', size = 15, labelpad=15)
        if yunit is 'phase':
            if self.phase is not None:
                nyticks = 5
                ax.set_yticks(np.linspace(0, self.nobs, nyticks))
                ax.set_yticklabels([ "{:.2f}".format(phase) for phase in np.linspace(self.phase[0], self.phase[-1], nyticks)])
                ax.tick_params(labelsize=12)
                ax.set_ylabel('Orbital phase', size=15)
            else:
                raise ValueError('Orbital phase has not been defined.')
        # Create a title
        ax.set_title(figtitle, fontsize=18, pad=10)
        
        # Create colorbar
        if add_cbar:
            cbar = Colorbar(ax = axes[-1], mappable = im, orientation = 'horizontal', ticklocation = 'bottom', extend='both')
            cbar.set_label(r'Flux (arbitrary units)', fontsize=15)
            cbar.ax.tick_params(labelsize=12)
        
            if return_cbar:
                return (axes, cbar)
            else:
                return axes
        else:
            return ax
        
    def inject(self, model, bbstar, rplanet, rstar, alpha=1.):
        """Inject model into spectral order."""
        model_inj = (model/bbstar)*(rplanet/rstar)**2
        data_inj = self.data * (1. + (alpha-1)*model_inj)
        return data_inj, model_inj
    
    def clip_edges(self, xmin, xmax):
        """Clip off the detector edges.
        
        xmin: leftmost detector pixel
        xmax: rightmost detector pixel
        """
        edges_mask = np.logical_or(self.wavsolution[0] < xmin, self.wavsolution[0] > xmax)
        self.data = self.data[:,~edges_mask]
        self.mask = self.mask[:,~edges_mask]
        self.wavsolution = self.wavsolution[:,~edges_mask]
        self.error = self.error[:,~edges_mask]
        self.nobs, self.nx = self.data.shape
            
    def detrend_brogi_line_19(self, dirout, polydeg=2, sigma_threshold=3,
                              norm_npoints=100, plot=True):
        """"""
        # Step 3 BL19: normalize spectrum
        data_n = np.zeros(self.data.shape)
        for n in range(self.nobs):
            spec = self.data[n, :]
            brightest_n_points = np.sort(spec)[-norm_npoints:]
            data_n[n, :] = spec/np.median(brightest_n_points)
        
        # Step 4 BL19: fit each exposure to mean spectrum
        spec_mean = np.mean(data_n, axis=0)
        fit_1 = np.zeros(data_n.shape)
        for n in range(self.nobs):
            coefs = np.polyfit(x=spec_mean, y=data_n[n, :], deg=polydeg)
            fit_1[n, :] = np.polyval(coefs, spec_mean)
        residual_1 = data_n/fit_1

        # Step 5 BL19: fit to lightcurve of each spectral channel
        frames = np.arange(1, self.nobs+1)
        fit_2 = np.zeros(self.data.shape)
        for m in range(self.nx):
            coefs = np.polyfit(x=frames, y=residual_1[:, m], deg=polydeg)
            fit_2[:, m] = np.polyval(coefs, frames)
        data_detrended = residual_1/fit_2
        
        # Step 6: masking bad columns
        cols_std = np.std(data_detrended, axis=0)/np.std(data_detrended)
        mask_detrended = np.broadcast_to(np.array(cols_std > sigma_threshold, dtype=int), self.data.shape)
        
        if plot:
            ax = self.plot(figtitle='Before detrending')
            fname = 'bl19_detrending_plot0_order_{}.png'.format(self.norder)
            plt.savefig(os.path.join(dirout, fname))
            plt.close()
            
            ax = self.plot(data=data_n, figtitle='Normalized by median of rows')
            fname = 'bl19_detrending_plot1_order_{}.png'.format(self.norder)
            plt.savefig(os.path.join(dirout, fname))
            plt.close()

            ax = self.plot(data=np.broadcast_to(spec_mean, self.data.shape), figtitle='Mean spectrum')
            fname = 'bl19_detrending_plot2_order_{}.png'.format(self.norder)
            plt.savefig(os.path.join(dirout, fname))
            plt.close()

            ax = self.plot(data=fit_1, figtitle='Fitted mean spectrum')
            fname = 'bl19_detrending_plot3_order_{}.png'.format(self.norder)
            plt.savefig(os.path.join(dirout, fname))
            plt.close()

            self.plot(data=residual_1, figtitle='Data / fitted mean spectrum')
            fname = 'bl19_detrending_plot4_order_{}.png'.format(self.norder)
            plt.savefig(os.path.join(dirout, fname))
            plt.close()

            ax = self.plot(data=fit_2, figtitle='Fitted lightcurves')
            fname = 'bl19_detrending_plot5_order_{}.png'.format(self.norder)
            plt.savefig(os.path.join(dirout, fname))
            plt.close()

            self.plot(data=data_detrended, apply_mask=True, mask=mask_detrended, figtitle='After Brogi & Line (19) detrending (masked)')
            fname = 'bl19_detrending_plot6_order_{}.png'.format(self.norder)
            plt.savefig(os.path.join(dirout, fname))
            plt.close()
        
        return data_detrended, mask_detrended
    
    def new_mask(self, threshold, envelope_deg=3, mode='sigma'):
        """Make a mask based on a threshold and noisy columns."""
        self.mask = get_mask(self.data, threshold, envelope_deg, mode)
        
    def __repr__(self):
        return 'SpectralOrder({}, {}, {})'.format(self.norder, self.obsdate, self.target)
    
    def align(self, mask, plot=True, osr=2, shiftmax=5, dirout=None, template_mode='first'):
        """"""
        # use first spectrum as template
        if template_mode == 'first':
            TEMPLATE_INDEX = 0
            template = self.data[TEMPLATE_INDEX,:]
        elif template_mode == 'median':
            template = np.median(self.data, axis=0)

        # define shifts
        shifts = np.linspace(-shiftmax, shiftmax, osr*shiftmax*2+1)

        # get cross-correlation matrix
        data_n = self.data / self.data_normalized()
        ccmatrix = calc_ccmatrix(data_n, template, shifts, mask)
        ccmatrix = np.abs(ccmatrix) # don't allow for negative cross-correlation

        # create grid for plot
        if plot:
            plt.figure(figsize=(16.8, 10))
            gx, gy = int(np.ceil(np.sqrt(self.nobs))), int(np.ceil(np.sqrt(self.nobs)))
            gs = gridspec.GridSpec(gx, gy)
            axes = []
            for n in range(gx):
                for m in range(gy):
                    axes.append(plt.subplot(gs[n, m]))
            for n in range(len(axes)):
                axes[n].set_yticks([])
                if n+1 > self.nobs:
                    axes[n].axis('off')

        # allocate memory for drift of each observed spectrum
        drift = np.zeros(self.nobs)
        if template_mode == 'first':
            istart = 1
        elif template_mode == 'median':
            istart = 0
        
        for i in range(istart, self.nobs):

            # define initial guess for gaussian
            y = ccmatrix[i,:]
            xos = np.linspace(shifts[0], shifts[-1], len(shifts))
            amp0 = np.max(y)
            mu0 = xos[np.argmax(y)]
            sigma0 = 1.
            y0 = 0.
            p0 = [amp0, mu0, sigma0, y0]

            # best fit
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    coefs, var_matrix = curve_fit(gaussian, shifts, y, p0=p0)
                except (RuntimeError, OptimizeWarning) as e:
                    coefs = p0 # set to initial guess
            drift[i] = coefs[1]

            if plot:
                # plot best fit
                axes[i].scatter(shifts, y, color='k', s=5, label='CC')
                axes[i].plot(xos, gaussian(xos, *coefs), color=TABLEAU20[6])
                axes[i].axvline(drift[i], color=TABLEAU20[6], ls='--')
                axes[i].set_ylim(y.min(),y.max())

        # outlier correction
        x = np.arange(len(drift))
        valid = (np.abs(drift) <= shiftmax)
        drift = np.interp(x, x[valid], drift[valid])

        if plot:
            if dirout is not None:
                # plot previous figure
                fname = 'alignment_fit_order_{}'.format(self.norder)+'.png'
                pathout = os.path.join(dirout, fname)
                plt.savefig(pathout, dpi=250)
                plt.close()
            else:
                plt.show()

            # second figure
            plt.figure()

            # some plot parameters
            obsid = np.arange(0, self.nobs)
            aspect = ((shiftmax*2+1)/ccmatrix.shape[0])

            ax = plt.imshow(ccmatrix, aspect=aspect,
                            extent=[shifts[0]-0.5, shifts[-1]+0.5,
                                    -0.5, self.nobs-0.5], cmap='Blues_r')
            plt.plot(drift[::-1], obsid, color='k', lw=0.5, label='Drift')
            cbar = plt.colorbar()
            cbar.set_label('CC', size=15)
            plt.title('CC Matrix (order={})'.format(self.norder), size=15)
            plt.xlabel('Shift [pixel]', size=15)
            plt.ylabel('# observation', size=15)
            leg = plt.legend(frameon=False)

            if dirout is not None:
                fname = 'cc_matrix_order_{}'.format(self.norder)
                pathout = os.path.join(dirout, fname)
                plt.savefig(pathout, dpi=250)
                plt.close()
            else:
                plt.show()

        data_aligned = realign(self.data, drift=drift)
        return drift, data_aligned
    
    def data_ma(self, data=None):
        if data is None:
            data = self.data
        return ma.masked_array(data, self.mask)
    
    def data_normalized(self, data=None):
        if data is None:
            data = self.data
        
        if self.mask is None:
            return np.broadcast_to(np.median(data, axis=1), data.shape)
        else:
            return np.broadcast_to(ma.median(self.data_ma(data), axis=1)[:,np.newaxis].data, data.shape)
            
    def data_column_mean_subtracted(self, data=None):
        if data is None:
            data = self.data
        
        if self.mask is None:
            return np.broadcast_to(np.mean(self.data, axis=0)[np.newaxis,:], data.shape)
        else:
            return np.broadcast_to(ma.mean(self.data_ma(data), axis=0)[np.newaxis,:].data, data.shape)
        
    def estimate_error(self, axis=0):
        """Estimate the individual error as the standard devation along a 
        specified axis."""
        if axis == 0:
            return np.broadcast_to(np.std(self.data, axis=0)[np.newaxis,:], self.data.shape)
        elif axis == 1:
            return np.broadcast_to(np.std(self.data, axis=1)[:,np.newaxis], self.data.shape)
        else:
            raise ValueError('Axis should be 0 or 1.')
        
        
def plot_eigenvalues(s):
    """
    Description:
        Create a plot of the eigenvalues.
    Input:
        s - list of eigenvalues
        outputfolder - save plot here
    """
    
    fig, ax = plt.subplots(figsize = (8, 5))
    plt.step(np.arange(len(s)), s, where='post', lw = 1.5)
    plt.xlabel('k', size = 15)
    plt.ylabel('log Eigenvalue', size = 15)
    plt.xlim(0, len(s)-1)
    plt.ylim(0, s[1])
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.show()

def pca_detrending(data, k, return_full=False):
    """
    Singular Value Decomposition of data using k eigenvectors.
    """

    U, s, V = np.linalg.svd(data, full_matrices=False)

    Uk = U[:,0:k]
    Vk = V[0:k,:]
    sk = s[0:k]
    Sk = np.diag(sk)
    
    model = np.dot(np.dot(Uk, Sk), Vk)
    residual = data - model
    
    if return_full:
        return model, residual, U, s, V
    else:
        return residual
    
def sliding_window_iter(iterable, size):
    """..."""
    iterable = iter(iterable)
    window = deque(islice(iterable, size), maxlen=size)
    for item in iterable:
        yield tuple(window)
        window.append(item)
    if window:  
        # needed because if iterable was already empty before the`for`,
        # then the window would be yielded twice.
        yield tuple(window)
        
        
def apply_highpass_filter(data, freq_cutoff, **kwargs):
    """Return high pass filtered data."""
    filter_order = kwargs.pop('N', 6)
    b, a = signal.butter(N=filter_order, Wn=freq_cutoff, btype='highpass', output='ba', fs=1, **kwargs)
    
    # when applying to the synthetic template
    if data.ndim == 1:
        data_f = signal.filtfilt(b, a, data)
        
    # when applying to the spectral time series data
    elif data.ndim == 2:
        data_f = np.zeros(data.shape)
        for n in range(data.shape[0]):
            data_f[n,:] = signal.filtfilt(b, a, data[n,:])
    
    else:
        raise ValueError('Array has too many dimensions.')
    
    return data_f

def butter_bandpass(freq_cutoff, **kwargs):
    """Return high pass filtered data."""
    filter_order = kwargs.pop('N', 6)
    b, a = signal.butter(N=filter_order, Wn=freq_cutoff, btype='highpass', output='ba', fs=1, **kwargs)
    return b, a

def butter_bandpass_filter(data, b, a):
    """Butterworth highpassfilter."""
    # when applying to the synthetic template dim=1
    if data.ndim == 1:
        data_f = signal.filtfilt(b, a, data)
    
    # when applying to the spectral time series data
    elif data.ndim == 2:
        data_f = np.zeros(data.shape)
        for n in range(data.shape[0]):
            data_f[n,:] = signal.filtfilt(b, a, data[n,:])   
    else:
        raise ValueError('Array has too few/many dimensions.')
    return data_f

class TemplateOrder:
    def __init__(self, data, wav, R_values, norder=None, targetname='', obsdate=None):
        """"""
        self.data = data
        self.nx, self.nframes = data.shape
        self.wavegrid = wav
        self.R_values = R_values
        self.norder = norder
        self.targetname = targetname
        self.obsdate = obsdate
    
    def __repr__(self):
        """"""
        return 'TemplateOrder({}, {}, {})'.format(self.norder, self.obsdate, self.targetname)
    
    @classmethod
    def load(cls, f, data_only=False):
        """"""
        data = fits.getdata(f+'.fits')
        if not data_only:
            wavegrid = np.load(f+'_wavegrid.npy')
            R_values = np.load(f+'_R_values.npy')
            with open(f+'_kwargs.pickle', 'rb') as handle:
                d = pickle.load(handle)
                norder = d['norder']
                targetname = d['targetname']
                obsdate = d['obsdate']
        return cls(data=data, wav=wavegrid, R_values=R_values, norder=norder, targetname=targetname, obsdate=obsdate)

    def save(self, f, data_only=False):
        fits.writeto(f+'.fits', self.data, overwrite=True)
        if not data_only:
            d = {
                'norder' : self.norder,
                'targetname' : self.targetname,
                'obsdate' : self.obsdate
            }
            np.save(f+'_wavegrid.npy', self.wavegrid)
            np.save(f+'_R_values.npy', self.R_values)
            with open(f+'_kwargs.pickle', 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def plot(self):
        fig = plt.figure(figsize=(11,5))
        gs = gridspec.GridSpec(1, 2)
        ax0 = plt.subplot(gs[0:1,0:1])
        ax1 = plt.subplot(gs[0:1,1:2])
        wavmin, wavmax = self.wavegrid[0], self.wavegrid[-1]

        ax0.plot(self.R_values/1e4, np.arange(1, self.nframes+1))
        xmin, xmax = (self.R_values.min()/1e4, self.R_values.max()/1e4)
        if not xmin == xmax:
            ax0.set_xlim(xmin, xmax) # this avoids UserWarning by matplotlib in case all values are set to the same value
        ax0.set_ylim(1, self.nframes)
        ax0.set_title('Measured \nspectral resolution')
        ax0.set_xlabel('R / 10,000')
        ax0.set_ylabel('# frame')

        ax1.imshow(self.data.T, origin='bottom',
                   extent=[wavmin, wavmax, 1, self.nframes], aspect=(wavmax-wavmin)/(self.nframes-1), cmap='inferno')
        ax1.set_title('Convolved templates')
        ax1.set_xlabel('Wavelength [micron]')
        return fig, (ax0, ax1)
