import os
import pickle
import numpy
from astropy.io import fits
import scipy
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.interpolate as interpolate
from matplotlib import gridspec

import sys
PYTHON_VERSION = sys.version_info[0]
base = os.path.abspath("../lib/ceres")+'/'
sys.path.append(base+"utils/Correlation")
sys.path.append(base+"utils/GLOBALutils")
sys.path.append(base+"utils/OptExtract")

from .constants import TABLEAU20, ARIES_NX, ARIES_NY
from .preprocessing import plot_image
from .preprocessing import is_flat, is_science

EchelleTrace = namedtuple('EchelleTrace', ['order', 'x', 'y'])

def unroll(trace):
    """Return unrolled Echelle trace."""
    distance = lambda x1, y1, x2, y2: np.sqrt((x2-x1)**2+(y2-y1)**2)
    arclength = np.cumsum([distance(trace.x[n], trace.y[n], \
                                    trace.x[n+1], trace.y[n+1]) \
                           for n in range(0, len(trace.x)-1)])
    x_unrolled = np.insert(arclength, 0, 0.)
    y_unrolled = np.full(trace.y.shape, trace.y[0])
    return x_unrolled, y_unrolled

def flatten_traces(traces):
    """Return flattend list of EchelleTrace objects."""
    xf = np.concatenate([trace.x for trace in traces])
    yf = np.concatenate([trace.y for trace in traces])
    return xf, yf

def get_unrolled_traces(traces):
    """Return unrolled list of EchelleTrace objects."""
    unrolled_traces = []
    for norder, trace in enumerate(traces):
        x_unrolled, y_unrolled = unroll(trace)
        unrolled_trace = EchelleTrace(x=x_unrolled, y=y_unrolled, order=norder)
        unrolled_traces.append(unrolled_trace)
    return unrolled_traces


class EchelleTraces:
    def __init__(self, coefs_all, imgshape=(ARIES_NX, ARIES_NY)):
        """Initialize Echele traces."""
        self.coefs_all = coefs_all
        self.norders, self.traces_degree = coefs_all.shape
        self.imgshape = imgshape # Default is ARIES detector's shape
        
        # Initialize traces
        self.traces = []
        x = np.arange(1, ARIES_NX+1)
        for norder, coefs in enumerate(self.coefs_all, 1):
            y = scipy.polyval(coefs, x)
            is_valid = (0 <= y) * (y <= x.max())
            trace = EchelleTrace(x=x[is_valid], y=y[is_valid], order=norder)
            self.traces.append(trace)
        
    @classmethod
    def load(cls, fpath):
        """Load traces from pickle file."""
        if PYTHON_VERSION == 3:
            trace = pickle.load(open(fpath, 'rb'), encoding='bytes')
            coefs_all = np.array(trace[b'coefs_all'])
            return cls(coefs_all)
        elif PYTHON_VERSION == 2:
            trace = pickle.load(open(fpath))
            coefs_all = np.array(trace['coefs_all'])
            return cls(coefs_all)
    
    def save(self, fpath):
        """Save traces to pickle file."""
        if PYTHON_VERSION == 3:
            trace_dict = {
                b'coefs_all' : self.coefs_all,
                b'norders' : self.norders
            }
            pickle.dump(trace_dict, open(fpath, 'wb'), protocol=2)
        elif PYTHON_VERSION == 2:
            trace_dict = {
                'coefs_all' : self.coefs_all,
                'norders' : self.norders
            }
            pickle.dump(trace_dict, open(fpath, 'w'))
            
    def __repr__(self):
        return "EchelleTraces(coefs_all = {})".format(self.coefs_all)

def plot_1d_cut_with_traces(img, coefs_all, column=100):
    """Plot a 1D cut of an image for a specified column with the trace centers."""
    median = np.median(img)
    sigma = np.std(img)
    
    vmin = median - 1 * sigma
    vmax = median + 5 * sigma

    npixels = img.shape[0]
    xrange = np.arange(1, npixels+1)

    # Get the center of the traces for this order
    trace_centers = []
    for coefs in coefs_all:
        center = scipy.polyval(coefs, column) + 1
        is_valid = (0 <= center) * (center <= xrange.max())
        if is_valid:
            trace_centers.append(center)
            
    fig, ax = plt.subplots(figsize=(7.5,7.5))
    ax.plot(xrange, img[:,column], color='k')
    for center in trace_centers:
        ax.axvline(center, color='r', lw=0.5)
    plt.xlim(xrange.min(), xrange.max())
    plt.xlabel('Y', size=15)
    plt.ylabel('Counts', size=15)
    
def plot_img_with_traces(img, coefs_all):
    """Plot an image and overlay the Echelle traces."""
    median = np.median(img)
    sigma = np.std(img)
    
    vmin = median - 1 * sigma
    vmax = median + 5 * sigma
    
    fig, ax = plt.subplots(figsize=(7.5,7.5))
    plt.imshow(img, cmap='Greys_r', vmin=vmin, vmax=vmax)
    plt.colorbar()

    npixels = img.shape[0]
    xrange = np.arange(1, npixels)
    for coefs in coefs_all:
        trace = scipy.polyval(coefs, xrange)
        is_valid = (0 <= trace) * (trace <= xrange.max())
        plt.plot(xrange[is_valid], trace[is_valid], color='r', lw=0.5)
        
    plt.xlabel('X', size=14)
    plt.ylabel('Y', size=14)
    
# def plot_traces(img, coefs_all, ax=None, color=None, yoffset=0, **kwargs):
#     """Plot the Echelle traces of an image."""
#     # Create figure if ax not specified
#     if ax is None:
#         fig, ax = plt.subplots()
    
#     # Get xrange
#     nx = img.shape[0]
#     xrange = np.arange(1, nx)
#     ncoefs = len(coefs_all)
    
#     # Define colors if not specified
#     DEFAULT_TRACE_COLOR = 'r'
#     if color is None:
#         colors = [DEFAULT_TRACE_COLOR for n in range(ncoefs)]
#     if type(color) is int:
#         print(color)
#         colors = [color for n in range(ncoefs)]
    
#     # Plot traces
#     for n, coefs in enumerate(coefs_all):
#         trace = scipy.polyval(coefs, xrange)
#         is_valid = (0 <= trace) * (trace <= xrange.max())
#         ax.plot(xrange[is_valid], trace[is_valid]+yoffset,
#                 color=colors[n], **kwargs)
#     return ax    

def plot_traces(img, coefs_all, ax=None, colors=None, yoffset=0, **kwargs):
    """Plot the Echelle traces of an image."""
    # Create figure if ax not specified
    if ax is None:
        fig, ax = plt.subplots()
    
    # Get xrange
    nx = img.shape[0]
    xrange = np.arange(1, nx)
    ncoefs = len(coefs_all)
    
    # Define colors if not specified
    DEFAULT_TRACE_COLOR = 'r'
    if colors is None:
        colors = [DEFAULT_TRACE_COLOR for n in range(ncoefs)]
    if type(colors) is str:
        colors = [colors for n in range(ncoefs)]
    
    # Plot traces
    for n, coefs in enumerate(coefs_all):
        trace = scipy.polyval(coefs, xrange)
        is_valid = (0 <= trace) * (trace <= xrange.max())
        ax.plot(xrange[is_valid], trace[is_valid]+yoffset,
                color=colors[n], **kwargs)
    return ax

def load_refcoefs(dirin_traces, fnames_flat):
    """Get al the reference trace coeficients 
    of all the flats in a directory."""
    refcoefs_all_flats = []
    for flat in fnames_flat:
        trace_path = os.path.join(dirin_traces, 'trace_'+flat[:-5]+'.pkl')
        reftrace_dict = pickle.load(open(trace_path))
        refcoefs_all = reftrace_dict['coefs_all']
        refcoefs_all_flats.append(refcoefs_all)
    return refcoefs_all_flats

def calc_shift_and_newcoefs(img, refcoefs_all_flats):
    """Get the shifts and updated coefiicients of 
    the science frames by retracing."""
    import Marsh
    from GLOBALutils import retrace

    shifts = []
    new_coefs_all_flats = []
    for refcoefs_all in refcoefs_all_flats:
        new_coefs_all, shift = retrace(dat=img, c_all=refcoefs_all)
        shifts.append(shift)
        new_coefs_all_flats.append(new_coefs_all)
    return shifts, new_coefs_all_flats


class EchelleImageTransformer:
    """Transformer to warp/dewarp Echelle images."""
    def __init__(self, traces, newshape):
        # transformed coordinate grid
        self.newshape = newshape
        self.xt = np.arange(1, newshape[0]+1)
        self.yt = np.arange(1, newshape[1]+1)
        self.yoffset = traces.imgshape[0] - newshape[0]
        
        # original coordinate grid
        self.x = np.arange(1, traces.imgshape[0]+1)
        self.y = np.arange(1, traces.imgshape[1]+1)
        self.xnew, self.ynew = self._get_newcoords(traces)
        
    def _interp_coefs_all(self, traces):
        """Return interpolated trace coeficients."""
        y0_all = np.array([trace.y[0] \
                          for n, trace in enumerate(traces.traces)])
        yt = np.arange(1, self.newshape[0]+1) + self.yoffset
        coefs_interp = np.zeros(shape=(self.newshape[0], traces.traces_degree))
        for degree in range(traces.traces_degree):
            interp_coef = interpolate.interp1d(x=y0_all,
                                   y=traces.coefs_all[:,degree],
                                   fill_value="extrapolate")
            coefs_interp[:,degree] = interp_coef(yt)
        return coefs_interp
            
    def _interp_traces(self, coefs_interp):
        """Return interpolated traces."""
        traces_interp = []
        xt = np.arange(1, self.newshape[1]+1)
        nrows = coefs_interp.shape[0]
        coefs_all = (coefs_interp[n,:] for n in range(nrows))
        for coefs in coefs_all:
            trace = EchelleTrace(x=self.x, y=scipy.polyval(coefs, self.x), order=None)
            x_unrolled, _ = unroll(trace)
            convert_to_xnew = interpolate.interp1d(x_unrolled, self.x, fill_value='extrapolate')
            
            xnew = convert_to_xnew(xt)
            ynew = scipy.polyval(coefs, xnew)
            trace = EchelleTrace(x=xnew, y=ynew, order=None)
            traces_interp.append(trace)
        return traces_interp
    
    def _get_newcoords(self, traces):
        """Return new x, y coordinates of dewarped grid."""
        coefs_interp = self._interp_coefs_all(traces)
        traces_interp = self._interp_traces(coefs_interp)
        xnew, ynew = flatten_traces(traces_interp)
        return xnew, ynew
        
    def dewarp(self, img):
        """Return dewarped image."""
        interp_img = interpolate.RegularGridInterpolator((self.x, self.y), img, fill_value=np.nan, bounds_error=False)
        points = list(zip(self.ynew, self.xnew))
        img_dewarped = np.array(interp_img(points)).reshape(self.newshape)
        return img_dewarped
    
    def warp(self, img):
        """Return warped image."""
        points = list(zip(self.ynew, self.xnew))
        values = img.flatten()
        xx, yy = np.meshgrid(self.x, self.y)
        img_warped = interpolate.griddata(points, values, (xx, yy), method='linear').T
        return img_warped
    
    @staticmethod
    def plot_result(img_before, img_after, vmin=None, vmax=None, cmap='Greys_r', **kwargs):
        """Plot before/after transformation image."""
        gs = gridspec.GridSpec(1,2, width_ratios=[1.,1.], wspace=0)
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1])

        # plot original image
        ax1.imshow(img_before, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', **kwargs)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # trick to hide colorbar ax1, but use equal aspect ratios
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)   
        cax.axis('off')

        # plot dewarped image
        ax2 = plot_image(img_after, ax=ax2, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        ax2.set_ylabel('')
        ax2.set_xlabel('')
        
def get_flats_and_traces(dirin_flats, dirin_traces):
    """Return list of flats and their Echelle Traces from an inputdir."""
    flats = []
    flats_traces = []
    STEM = slice(0, -5, 1) #no .fits extension
    flats_fname = [fname[STEM] for fname in os.listdir(dirin_flats) if is_flat(fname)]
    for fname in flats_fname:
        flats.append(fits.getdata(os.path.join(dirin_flats, fname+'.fits')))
        flats_traces.append(EchelleTraces.load(os.path.join(dirin_traces, 'trace_'+fname+'.pkl')))
    flats = np.array(flats)
    return flats, flats_traces

def get_master_traces(traces_all, method='median'):
    """Return master image EchelleTraces from a list of traces."""
    traces_coefs_all = np.array([traces.coefs_all for traces in traces_all])
    if method == 'median':
        master_coefs_all = np.median(traces_coefs_all, axis=0)
    elif method == 'mean':
        master_coefs_all = np.mean(imgs, axis=0)
    else:
        raise ValueError('Invalid method {}.'.format(method))
    master_traces = EchelleTraces(coefs_all = master_coefs_all)
    return master_traces

def load_traces(dirin, fnames):
    traces = []
    for fname in fnames:
        traces.append(EchelleTraces.load(os.path.join(dirin, fname+'.pkl')))
    return traces