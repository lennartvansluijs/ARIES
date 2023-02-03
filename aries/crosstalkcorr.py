import csv
import os
import shutil
import subprocess
import numpy as np
from operator import itemgetter
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product
from collections import namedtuple

from .preprocessing import get_imgs_in_dir
from .preprocessing import get_keyword_from_headers_in_dir
from .utils import cd


DEFAULT_CORQUAD_COEFS = {
    'a0' : -3.343291E-4,
    'a1' : 7.858745E-6,
    'a2' : -7.819456E-8,
    'a3' : 3.687237E-10,
    'a4' : -6.67760E-13,
    'kern0' : -0.004,
    'kern1' : -0.005,
    'kern2' : -0.001
}


def locate_coolpixels(hotx, hoty):
    """Return coolpixels in other three quadrants.

    Due to crosstalk, a hot pixel in one quadrant of the array
    causes cool pixels at the same location in the other three quadrants.

    Args
        hotx, hoty -- position of the hotpixel

    Return
        coolpixels -- len=3 tuple of len=2 tuples, ((x1, y1), ..., (x3, y3))
    """

    in_quadrant_1 = (hotx < 512) and (hoty < 512)
    in_quadrant_2 = (hotx > 512) and (hoty < 512)
    in_quadrant_3 = (hotx > 512) and (hoty > 512)
    in_quadrant_4 = (hotx < 512) and (hoty > 512)

    coolpixels = lambda xadd, yadd: \
        ((hotx + xadd, hoty),
         (hotx, hoty + yadd),
         (hotx + xadd, hoty + yadd))

    if in_quadrant_1:
        return coolpixels(512, 512)
    if in_quadrant_2:
        return coolpixels(-512, 512)
    if in_quadrant_3:
        return coolpixels(-512, -512)
    if in_quadrant_4:
        return coolpixels(512, -512)

def get_median(imgs):
    """Return median of images."""
    return np.median(imgs, axis = 0)

def calc_corquad_coefs(hotx, hoty, img, **settings):
    """Return the corquad coefficients and standard deviations
    from the hotpixels in an image.

    Args
        hotx -- 1d array, x-coordinates of hot pixels
        hoty -- 1d array, y-coordinates of hot pixels
        img    -- 2d array, the image

    Return:
        corquad_coefs -- a len=3 tuple with the 0th, 1st, and 2nd corquad
                         coefficients
        corquad_coefs_std -- a len=3 tuple with the corresponding standard
                             deviation
    """
    silent = settings.pop('silent', False)

    cool = map(locate_coolpixels, hotx, hoty)
    kern0 = [] # allocate memory for arrays used
    kern1 = [] # to calculate the corquad coefs
    kern2 = []
    for point in zip(hoty, hotx, cool):
        hot0 = img[point[0], point[1] - 1]
        hot1 = img[point[0], point[1]]
        hot2 = img[point[0], point[1] + 1]
        hot3 = img[point[0], point[1] + 2]
        # print('hots: ', hot0, hot1, hot2, hot3)
        for p in point[2]:
            cold1 = img[p[0], p[1]]
            cold2 = img[p[0], p[1] + 1]
            cold3 = img[p[0], p[1] + 2]
            # print('colds: ', cold1, cold2, cold3)
            median=np.median(img[p[0]-10:p[0]+10, p[1]-10:p[1]+10])

            k0=(median-cold1)/hot1
            k1=((median-cold2)-(hot2*k0))/hot1
            k2=((median-cold3)-(hot3*k0)-(hot2*k1))/hot1

            kern0.append(k0)
            kern1.append(k1)
            kern2.append(k2)

    if not silent:
        print('Estimate of corquad kernel coefs: ')
        print('kern 1: ',np.mean(kern0), ' +/- ',np.std(kern0))
        print('kern 2: ',np.mean(kern1), ' +/- ',np.std(kern1))
        print('kern 3: ',np.mean(kern2), ' +/- ',np.std(kern2))

    corquad_coefs = (np.mean(kern0), np.mean(kern1), np.mean(kern2))
    corquad_coefs_std = (np.std(kern0), np.std(kern1), np.std(kern2))
    return corquad_coefs, corquad_coefs_std

def find_hot_pixels(img, max_pixels = 100, sigma = 5.):
    """Return a list of hot pixels.

    Args
        img --2d array, the image with hot pixels
        max_pixels -- int, maximum hot pixels to find
        sigma -- float, hot pixel threshold

    Return
        tuple -- a len=2 tuple of numpy arrays, The zeroeth
                element is an array of x-coordinates, the
                first element is an array of y-coordinates.
    """
    threshold = np.median(img) + sigma * np.std(img)
    hot = np.where(img > threshold)

    # Exclude the edges of the images.
    indsa=np.logical_and(hot[0]>11,hot[0]<502)
    indsb=np.logical_and(hot[0]>525,hot[0]<940)
    inds1=np.logical_or(indsa,indsb)
    indsc=np.logical_and(hot[1]>11,hot[1]<502)
    indsd=np.logical_and(hot[1]>525,hot[1]<940)
    inds2=np.logical_or(indsc,indsd)
    inds=np.logical_and(inds1,inds2)

    hotx = hot[1][inds]
    hoty = hot[0][inds]

    vals = np.array([img[yx[0],yx[1]] for yx in zip(hoty,hotx)])
    if len(vals) > max_pixels:
        big_inds = np.argsort(vals)[-max_pixels:]
    else:
        big_inds = np.argsort(vals)

    hotx=hotx[big_inds]
    hoty=hoty[big_inds]

    return hotx, hoty

def write_hot_pixels_to_file(hotx, hoty, fname = 'corquad_hotpix.txt'):
    """Write hot pixels to file."""
    with open('corquad_hotpix.txt', 'w') as f:
        f.write('hotx, hoty\n')
        for point in zip(hotx, hoty):
            f.write('%i %i\n' %point)

def del_files(flist):
    """Delete files in a list of files."""
    for f in flist:
        os.remove(f)

class Corquad:
    """Class to correct for the crosstalk between the IR arrays.

    EXAMPLE 1 -- RUN CORQUAD ON A DIRECTORY
        new_corquad_coefs = {
            'a0': -3.343291E-4,
            'a1': 7.858745E-6,
            'a2' : -7.819456E-8,
            'a3' : 3.687237E-10,
            'a4' : -6.67760E-13,
            'kern0' : -0.0032,
            'kern1': -0.0021,
            'kern2' : -0.0006
        }
        corquad = Corquad(outputdir = '/home/lennart/api/measure/output_pipeline/kelt7/corquad')
        corquad.coefs = new_corquad_coefs
        darks_dirname = '/home/lennart/api/measure/images/kelt7/darks'
        corquad.run(inputdir = darks_dirname)

    EXAMPLE 2 -- RUN CORQUAD AND PLOT RESULT
        corquad = Corquad(outputdir = '/home/lennart/api/measure/output_pipeline/kelt7/corquad')
        darks_dirname = '/home/lennart/api/measure/images/kelt7/darks'
        corquad.run(inputdir = darks_dirname)
        result = corquad.result(plot = True)
    """

    def __init__(self, outputdir, executable_path, **kwargs):
        """Initialize corquad attributes."""
        self.outputdir = outputdir
        self.executable_path = executable_path
        self.coefs = kwargs.pop('coefs', DEFAULT_CORQUAD_COEFS)

    @property
    def _dotcorquad_path(self):
        HOME = os.path.expanduser('~')
        return os.path.join(HOME, '.corquad')

    def _update_dot_corquad(self):
        """Update the .corquad file containing the corquad coefs."""
        with open(self._dotcorquad_path, 'w') as f:
            for coef, value in self.coefs.items():
                f.write('{}\t{}\n'.format(coef, value))

    @property
    def coefs(self):
        return self._coefs

    @coefs.setter
    def coefs(self, value):
        self._coefs = value
        self._update_dot_corquad()

    def _move_file_to_output_dir(self, fname):
        """Move the corquad output to a new output directory."""
        output_fname = 'q' + fname
        new_output_path = os.path.join(self.outputdir, fname)
        try:
            os.rename(output_fname, new_output_path)
        except FileNotFoundError:
            os.makedirs(self.outputdir)
            os.rename(output_fname, new_output_path)

    def save_coefs(self):
        """Save a copy of the .corquad file used for this run."""
        new_path = os.path.join(self.outputdir, 'coefs.txt')
        shutil.copy(self._dotcorquad_path, new_path)

    def run(self, inputdir, key=''):
        """Run corquad on all files in a user specified directory."""
        # The corquad executable works only on .fits files.
        # The output of the corquad executable have a filename with a 'q' prepended.
        # Therefore, 'good files' are (1) .fits and (2) don't start with a 'q'
        # and (optionally) (3) a key to specify the file type.
        is_good_file = lambda fname: fname.endswith('.fits') \
         and (fname[0] != 'q') and (key in fname)
        # Executable corquad-linux only works when used in the current working directory.
        with cd(inputdir):
            fnames = [fname for fname in os.listdir() if is_good_file(fname)]
            for fname in fnames:
                subprocess.call([self.executable_path, fname])
                self._move_file_to_output_dir(fname)
                self.save_coefs()

        # Save last used input directory
        self._last_inputdir = inputdir

    def _plot_result(self, stamp_before, stamp_after, stamp_diff,
                     std_before, std_after, **settings):
        """Plot the result of a run."""
        savefig = settings.pop('savefig', True)
        figname = settings.pop('figname', 'result')

        # Create a figure.
        fig, axes = plt.subplots(ncols=3, figsize = (16,5))
        fig.subplots_adjust(left=0.05, right=0.95)

        # No ticks along images
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # Set limits of images the same
        vmin = min((stamp_before.min(), stamp_after.min(), stamp_diff.min()))
        vmax = 1000

        # Plot image data
        kwargs = {'cmap' : 'Greys_r', 'vmin' : vmin, 'vmax' : vmax}
        im = axes[0].imshow(stamp_before, **kwargs)
        axes[1].imshow(stamp_diff, **kwargs)
        axes[2].imshow(stamp_after, **kwargs)

        # Create colorbars
        for ax in axes:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, extend='max')
            cbar.set_label('Counts', size = 15)

        axes[0].set_title('Uncorrected (std: {:.2f})'.format(std_before), size = 15)
        axes[1].set_title('Corequad model', size = 15)
        axes[2].set_title('Corrected (std: {:.2f})'.format(std_after), size = 15)

        if savefig:
            output_path = os.path.join(self.outputdir, figname)
            plt.savefig(output_path + '.pdf')
            plt.savefig(output_path + '.png', dpi = 250)
            plt.close()

    def result(self, refpixel = (117, 92), dw = 10, sigma = 5., **settings):
        """Return the standard deviation improvement of last run."""
        silent = settings.pop('silent', True)
        plot = settings.pop('plot', True)
        save = settings.pop('save', True)

        darks_before = get_imgs_in_dir(self._last_inputdir, key='dark')
        darks_after = get_imgs_in_dir(self.outputdir, key='dark')

        darks_median_before = get_median(darks_before)
        darks_median_after = get_median(darks_after)
        diff = darks_median_before - darks_median_after

        # Take out a small 'stamp' of each image.
        pixel = namedtuple('pixel', ['x', 'y'])
        refpixel = pixel(*refpixel)
        DX = slice(refpixel.x-dw,refpixel.x+dw,1)
        DY = slice(refpixel.y-dw,refpixel.y+dw,1)
        stamp_before = darks_median_before[DX,DY]
        stamp_after = darks_median_after[DX,DY]
        stamp_diff = diff[DX,DY]

        # Get the improvement
        threshold = np.mean(darks_before) + sigma * np.std(darks_before)
        MASK_HOTPIXELS = np.where(stamp_after < threshold)
        std_before = np.std(stamp_before[MASK_HOTPIXELS])
        threshold = np.mean(darks_after) + sigma * np.std(darks_after)
        MASK_HOTPIXELS = np.where(stamp_after < threshold)
        std_after = np.std(stamp_after[MASK_HOTPIXELS])
        std_diff = std_after - std_before

        if not silent:
            print('Result of last run:')
            print('Std before, std after, std diff: ',
                   std_before, std_after, std_diff)

        if plot:
            self._plot_result(stamp_before, stamp_after, stamp_diff,
                              std_before, std_after, **settings)

        if save:
            fpath = os.path.join(self.outputdir, 'result.txt')
            with open(fpath, 'w') as file:
                file.write('Std after corquad: ')
                file.write(str(std_after))

        return std_after

class CorquadFitter:
    """Fit for the best corquad coefs using a grid-based approach.

    EXAMPLE -- Find best corquad coefs

    settings = {
        'silent' : False,
        'save' : True,
        'plot' : True
    }

    darks_dirname = '/home/lennart/api/measure/images/kelt7/darks'
    output_dirname = '/home/lennart/api/measure/output_pipeline/kelt7'
    corquad_fitter = CorquadFitter(inputdir = darks_dirname,
                                   outputdir = output_dirname)
    corquad_coefs, corquad_coefs_std = corquad_fitter.estimate_coefs(**settings)

    ngrid = 2
    gridrange = [np.linspace(coef - std, coef + std, ngrid) \
                 for coef, std in zip(corquad_coefs, corquad_coefs_std)]
    corquad_fitter.gridrange = gridrange

    bestcoefs = corquad_fitter.run(**settings)"""

    def __init__(self, inputdir, outputdir, executable_path, **kwargs):
        """Initialize corquad fitter.

        Input
            inputdir -- str, name of input directory.
                             Should contain long exposure darks.
            outputdir -- str, output directory name.
            executable_path -- str, path to corquad executable.
            coefs -- dict, corquad coefs.
                     Use DEFAULT_CORQUAD_COEFS if not specififed.
        """
        self.inputdir = inputdir
        self.outputdir = os.path.join(outputdir)
        self.executable_path = executable_path
        self.gridrange = kwargs.pop('gridrange', None)

    def estimate_coefs(self, max_nhots = 100, sigma=5., **settings):
        """Return the estimated corquad coefs with the standard deviation.

        Args
            max_pixels -- int, maximum number of hot pixels to use
            min_exptime -- float, minimum exposure time. Only use long exposure
                           darks in the directory to estimate the coefs.

        Return
            corquad_coefs -- len=3 tuple, estimated corquad coefs
            corquad_coefs_std -- len=3 tuple, estimated corquad coefs std
        """
        darks = get_imgs_in_dir(self.inputdir, key='dark')
        exptimes = get_keyword_from_headers_in_dir('EXPTIME',
            self.inputdir, key='dark')
        # Used longest exposure darks
        LONGEST_EXPTIME = exptimes.max()
        good_darks = np.array([dark for dark, exptime in zip(darks, exptimes) \
                               if exptime >= LONGEST_EXPTIME])
        darks_median = get_median(good_darks)

        hotx, hoty = find_hot_pixels(darks_median, max_nhots, sigma)
        corquad_coefs, corquad_coefs_std = \
            calc_corquad_coefs(hotx, hoty, darks_median, **settings)
        return corquad_coefs, corquad_coefs_std

    def _make_outputdir(self):
        """Make the output directory."""
        if not os.path.isdir(self.outputdir):
            os.makedirs(self.outputdir)

    def _explore_grid(self, **settings):
        """Explore the grid."""
        silent = settings.pop('silent', False)

        corquad = Corquad(outputdir = self.outputdir,
                          executable_path = self.executable_path)
        grid = product(*self.gridrange)
        gridsize = len(self.gridrange[0]) * len(self.gridrange[1]) * len(self.gridrange[2])

        results = []
        coefs = {**DEFAULT_CORQUAD_COEFS}
        for n, kernels in enumerate(grid):
            if not silent:
                print(f'Run ({n+1}/{gridsize})')
            coefs['kern0'], coefs['kern1'], coefs['kern2'] = kernels
            corquad.coefs = coefs
            corquad.outputdir = os.path.join(self.outputdir, f'run{n+1}')
            corquad.run(self.inputdir, key='dark')
            std = corquad.result(silent=silent, **settings)
            results.append({'run' : n+1, 'std' : std, **coefs})

        headers = ['run', 'std', 'a0', 'a1', 'a2', 'a3', 'a4', 'kern0', 'kern1', 'kern2']
        fpath = os.path.join(self.outputdir, 'results.csv')
        with open(fpath, 'w') as f:
            f_csv = csv.DictWriter(f, headers)
            f_csv.writeheader()
            f_csv.writerows(results)

        return results

    def get_bestcoefs(self, results, **settings):
        """Return the best fit corquad coefs."""
        silent = settings.pop('silent', False)
        save = settings.pop('save', True)

        bestfit = min(results, key=itemgetter('std'))

        if not silent:
            print('\nCorquad fitter result:')
            print(f'Best run: {bestfit["run"]}')
            print(f'Best std: {bestfit["std"]}')
            print(f'Best kern0: {bestfit["kern0"]}')
            print(f'Best kern1: {bestfit["kern1"]}')
            print(f'Best kern2: {bestfit["kern2"]}')

        bestfit_coefs = {
            'a0' : DEFAULT_CORQUAD_COEFS['a0'],
            'a1' : DEFAULT_CORQUAD_COEFS['a1'],
            'a2' : DEFAULT_CORQUAD_COEFS['a2'],
            'a3' : DEFAULT_CORQUAD_COEFS['a3'],
            'a4' : DEFAULT_CORQUAD_COEFS['a4'],
            'kern0' : bestfit["kern0"],
            'kern1' : bestfit["kern1"],
            'kern2' : bestfit["kern2"]
        }

        if save:
            fpath = os.path.join(self.outputdir, 'bestfit_coefs.txt')
            with open(fpath, 'w') as f:
                for coef, value in bestfit_coefs.items():
                    f.write('{}\t{}\n'.format(coef, value))

        return bestfit_coefs

    def run(self, **settings):
        """Run the fitter."""
        self._make_outputdir()
        results = self._explore_grid(**settings)
        bestfit_coefs = self.get_bestcoefs(results, **settings)

        return bestfit_coefs

    def cleanup(self):
        """Cleanup by deleting all .fits files."""
        gridsize = len(self.gridrange[0]) * len(self.gridrange[1]) * len(self.gridrange[2])
        for n in range(1, gridsize):
            outputdir = os.path.join(self.outputdir, f'run{n+1}')
            fits_files = [os.path.join(outputdir, f) for f in os.listdir(outputdir) \
                          if f.endswith('.fits')]
            del_files(fits_files)

if __name__ == '__main__':
    pass
