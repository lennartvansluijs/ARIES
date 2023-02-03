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

DARK_SHORT_EXPTIME = 60.0
DARK_LONG_EXPTIME = 300.0

is_science = lambda fname: ('science' in fname)
is_flat = lambda fname: ('flat' in fname)
is_dark = lambda fname: ('dark' in fname)

def find_hots(img, sigma = 5.0):
    """Find all hotpixels in an image."""
    threshold = np.mean(img) + sigma * np.std(img)
    hots = (img > threshold)
    return hots

def correct_hots(imgs, hots):
    """Correct for the hot pixels."""
    for img in imgs:
        img[hots] = np.median(img)
    return imgs

def gen_hduls(dirname, key=''):
    """Yield hduls of .fits files in a directory."""
    files = os.listdir(dirname)
    is_right_file = lambda file: (file.endswith('.fits') and (key in file))
    file_paths = [os.path.join(dirname, f) for f in files \
                  if is_right_file(f)]
    for file_path in file_paths:
        with fits.open(file_path) as hdul:
            yield hdul

def get_imgs_in_dir(dirname, key=''):
    """Return a list of images in a directory."""
    hduls = gen_hduls(dirname, key=key)
    imgs = np.array([hdul[0].data for hdul in hduls])
    return imgs

def get_headers_in_dir(dirname, key=''):
    """Return a list of fits headers in a directory."""
    hduls = gen_hduls(dirname, key=key)
    headers = [hdul[0].header for hdul in hduls]
    return headers

def get_keyword_from_headers_in_dir(keyword, dirname, key=''):
    """Return a list of the keyword values from .fits files in a directory."""
    hduls = gen_hduls(dirname, key=key)
    keyword_values = np.array([hdul[0].header[keyword] for hdul in hduls])
    return keyword_values

def perform_simple_photometry(masters):
    """Return final image.

    Args
        master_imgs -- namedtuple, contains master flat/dark/science

    Return
        final_img -- np.array, final image after applying simple photometry
    """
    final_img = (masters.science - masters.dark) \
                / (masters.flat - masters.dark)
    return final_img

def get_center_of_flux(imgs):
    """Return a list with the center of fluxes for a list of images."""
    center_of_flux = np.array([measurements.center_of_mass(flat) \
                              for flat in flats])
    return center_of_flux

def convert_airmass_to_elevation(airmass):
    """Return elevations."""
    return 90. - np.arccos(1./airmass) * 180./np.pi

def get_elevation_from_fnames(dirname):
    """Return a list of elevations as specified in flats filenames."""
    fnames = os.listdir(dirname)
    ELEVATION = slice(4,6,1) # this part of the filename contains the elevation
    elevation = np.array([float(fname[ELEVATION]) for fname in fnames])
    return elevation

def unpack_positions(positions):
    """Unpack a list of tuples containing positions."""
    return map(np.array, zip(*positions)) # Equivalent to "return x, y"
                                          # or "return x, y, z"

def plot_center_of_flux_with_elevation(centers_of_flux, elevations):
    """Show a plot with centers of flux with colors representing elevation."""
    center_of_flux_x, center_of_flux_y = unpack_positions(centers_of_flux)
    sc = plt.scatter(center_of_flux_x,
                     center_of_flux_y,
                     c = elevations,
                     cmap = 'viridis')

    plt.xlabel('X center of flux [pix]', size = 13)
    plt.ylabel('Y center of flux [pix]', size = 13)

    cbar = plt.colorbar(sc)
    cbar.set_label(r'Elevation [$\degree$]', size = 13)

    plt.show()
    plt.close()

def get_master(imgs, method='median'):
    """Return master image from a list of images with equal exptime."""
    if method == 'median':
        master = np.median(imgs, axis=0)
    elif method == 'mean':
        master = np.mean(imgs, axis=0)
    else:
        raise ValueError('Invalid method {}.'.format(method))
    return master

def correct_for_dark_current(inputdir, outputdir, master_dark):
    """Correct all images in an input directory for the dark current."""
    imgs = get_imgs_in_dir(inputdir)
    imgs_exptime = get_keyword_from_headers_in_dir('EXPTIME', inputdir)

    # Correct for dark current
    # TO DO: 60 SEC / 300 SEC
    nflats = imgs.shape[0]
    imgs = imgs \
        - master_dark.reshape(1, 1024, 1024) \
        * imgs_exptime.reshape(nflats, 1, 1)

    # Save images with original header in the outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    files = [f for f in os.listdir(inputdir) if f.endswith('.fits')]
    headers = get_headers_in_dir(inputdir)
    for fname, data, header in zip(files, imgs, headers):
        fpath = os.path.join(outputdir, fname)
        fits.writeto(fpath, data, header,
                     output_verify="ignore", overwrite=True)

def plot_image(img, ax=None, cmap='Greys_r', vmin=None, vmax=None, return_cbar=False, origin='lower', **kwargs):
    """Plot an image."""

    # Get vmin and vmax
    IMG_MIN = np.nanmin(img)
    IMG_MAX = np.nanmax(img)    
    if vmin is None:
        vmin = IMG_MIN
    if vmax is None:
        vmax = IMG_MAX
    
    # Check if colorbar is extended
    if (IMG_MIN < vmin) and (IMG_MAX > vmax):
        extend = 'both'
    elif (IMG_MIN < vmin) and (IMG_MAX <= vmax):
        extend = 'min'
    elif (IMG_MIN >= vmin) and (IMG_MAX > vmax):
        extend = 'max'
    else:
        extend = 'neither'
    
    # Create new figure if axes not exist
    if ax is None:
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.05, right=0.95)
    
    # No ticks along images
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('X', size=15)
    ax.set_ylabel('Y', size=15)

    # Plot image data
    im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap, origin=origin, **kwargs)
    
    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, extend=extend)
    cbar.set_label('Counts', size=15)
    
    if return_cbar:
        return ax, cbar
    else:
        return ax
    
def fix_badpixels(img, badpixelmap, size=3, mode='point', func=np.median):
    """Return badpixel corrected image."""
    if mode == 'point':
        footing = make_point_footing(size)
    elif mode == 'crosshair':
        footing = make_crosshair_footing(size)
    else:
        raise ValueError('Invalid mode: {}.'.format(mode))
        
    nx, ny = img.shape
    padding = np.copy(size)
    img_padded = np.pad(img, padding, 'reflect')
    aperture = (footing == 1)
    
    hots = zip(*np.where(badpixelmap == 1))
    for x, y in hots:
        DX = slice(x-int(size/2)+padding, x+int(size/2)+padding+1, 1)
        DY = slice(y-int(size/2)+padding, y+int(size/2)+padding+1, 1)
        img_stamp = img_padded[DX, DY]
        fill_value = func(img_stamp[aperture])
        img_stamp[~aperture] = fill_value
    
    img_corr = img_padded[padding:-padding, padding:-padding]
    return img_corr

def make_crosshair_footing(size):
    """Return a footing in the shape of a crosshair."""
    if (size % 2 != 1) or (size < 5):
        raise ValueError('Invalid size. Size must be an odd number >= 5.')
        
    footing = np.ones((size, size))
    x0, y0 = int(size/2), int(size/2)
    
    offsets = [(0,0), (1,0), (2,0), (0,1), (-1,0), (0,-1)]
    for dx, dy in offsets:
        footing[y0+dy, x0+dx] = 0
    return footing

def make_point_footing(size):
    """Return a footing in the shape of a single point."""
    if (size % 2 != 1) or (size < 3):
        raise ValueError('Invalid size. Size must be an odd number >= 3.')
    
    footing = np.ones((size, size))
    x0, y0 = int(size/2), int(size/2)
    footing[x0, y0] = 0
    return footing

def make_badpixelmap(img, sigma=5.):
    """Return badpixel map of an image."""
    uplim = np.median(img) + sigma * np.std(img)
    lowlim = 0 # include negative pixels as badpixels
    badpixels = np.logical_or(img > uplim, img < lowlim)
    badpixelmap = np.array(badpixels, dtype='int')
    return badpixelmap

def fix_badpixels_with_medfilt(img, badpixelmap, medfilt_kernel_size=5):
    """Return hotpixel corrected image."""
    img_blurred = medfilt2d(img, kernel_size=medfilt_kernel_size)
    img_corr = np.copy(img)
    badpixels = np.where(badpixelmap == 1)
    img_corr[badpixels] = img_blurred[badpixels]
    return img_corr

def robust_polyfit(x, y, deg=3, sigma=5., return_outliers=False, return_full=False):
    """Return best-fit envelope model of this row."""
    converged = False
    outliers = np.full(len(y), False) # start with no outliers
    while not converged:
        coefs = scipy.polyfit(x[~outliers], y[~outliers], deg=deg)
        yfit = scipy.polyval(coefs, x)
        diff = y - yfit

        new_outliers = np.abs(diff) >= sigma * np.std(diff)
        if np.array_equal(outliers, new_outliers):
            converged = True
        elif np.all(new_outliers):
            converged = True
        else:
            outliers = new_outliers
    
    if return_outliers:
        return yfit, outliers
    if return_full:
        return yfit, coefs, outliers
    else:
        return yfit
    
def identify_badcolumns(img, sigma=5, deg=5):
    """Return list of indices of badcolumns in image."""
    smooth_model = robust_polyfit(x=np.arange(img.shape[1]),
                                  y=np.median(img, axis=0),
                                  sigma=sigma, deg=deg)
    residual = np.median(img, axis=0) - smooth_model
    badcolumns = np.where(residual < residual.mean() - sigma*residual.std())[0]
    return badcolumns

def correct_badcolumns(img, badcolumns):
    """Return image corrected for bad column."""
    img_corr = np.copy(img)
    xrange = np.arange(img.shape[1])
    for column in badcolumns:
        closest_neighbouring_columns = np.argsort(np.abs(xrange-column))[1:3]
        img_corr[:, column] = (img[:, closest_neighbouring_columns[0]] + \
                               img[:, closest_neighbouring_columns[1]])/2.
    return img_corr

def make_badpixelmap(img, sigma=5., kernel_size=25):
    """Return badpixel map of an image."""
    img_blurred = medfilt2d(img, kernel_size=kernel_size)
    residual = img - img_blurred

    uplim = sigma * np.std(residual)
    lowlim = -sigma * np.std(residual)
    badpixels = np.logical_or(residual < lowlim, residual > uplim)
    
    badpixelmap = np.array(badpixels, dtype='int')
    return badpixelmap

def fit_illumination_model(img, traces, yoffset, polydegree=7, aperture=25):
    """Return illumination model of a dewarped image."""
    x = np.arange(img.shape[0])
    y0_traces = np.array([trace.y[0] for trace in traces.traces])
    y0_traces_dewarped = y0_traces - yoffset
    
    illumination_model = np.full(img.shape, np.nan)
    
    for y0 in y0_traces_dewarped:
        aperture_window = np.arange(int(y0)-int(aperture/2.),
                                    int(y0)+int(aperture/2.)) - 1
        for i in aperture_window:
            row = img[i, :]
            is_valid = ~np.isnan(row)
            illumination_model[i,:][is_valid] = robust_polyfit(x[is_valid],
                                                               row[is_valid],
                                                               deg=polydegree)
    return illumination_model

def get_fits_fnames(dirin, key=''):
    """Return list of all fits files."""
    STEM = slice(0, -5, 1) #no .fits extension
    fnames = [fname[STEM] for fname in os.listdir(dirin) if (key in fname) and fname.endswith('.fits')]
    return fnames

def load_imgs(dirin, fnames):
    imgs = []
    for fname in fnames:
        imgs.append(fits.getdata(os.path.join(dirin, fname+'.fits')))
    return imgs

def fix_badpixels_science(img, badpixelmap):
    img_fixed = np.zeros(img.shape)
    x = np.arange(0, img.shape[0])
    for j in range(0, img.shape[0]-1):
        column = img[:, j]
        badpixels = np.array(badpixelmap[:, j], dtype='bool')
        img_fixed[:, j] = np.interp(x=np.arange(0, img.shape[0]), xp=x[~badpixels], fp=column[~badpixels])
    return img_fixed
