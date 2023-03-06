import os
from collections import namedtuple
import numpy as np
import numpy.polynomial.polynomial as polynomial
import matplotlib.pyplot as plt

from .constants import TABLEAU20
from .preprocessing import get_imgs_in_dir
from .preprocessing import get_elevation_from_fnames


Pixel = namedtuple('pixel', ['x','y'])


def convolve_with_kernel(x, kernel):
    """Return convolved array with kernel."""
    return np.convolve(x, kernel, 'same')

def get_grad(x):
    """Return the gradient of an array."""
    return np.abs(np.diff(x))

def get_argmaxs(x):
    """Return the local maxima of an array."""
    argmaxs = []
    prev_value = x[0]
    for i in range(1, len(x)-1):
        next_value = x[i+1]
        is_max = (prev_value <= x[i]) and (x[i] >= next_value)
        if is_max:
            argmaxs.append(i)
        prev_value = x[i]
    return np.array(argmaxs)

def pixel_to_ind(value):
    """Convert a pixel value to the index."""
    return value-1

def ind_to_pixel(value):
    """Convert a index value to a pixel."""
    return value+1

def gauss_func(x, mu=0., sigma=1.):
    """Return a Gaussian with a mean mu and standard deviation sigma."""
    return np.exp(-0.5*((x-mu)/sigma)**2)

def find_closest(edges, refpixel):
    """Return two edges closest to reference pixel."""
    distance = np.array([np.abs(edge-refpixel.y) for edge in edges])
    ind = np.argsort(distance)[:2]
    topy, boty = edges[np.max(ind)], edges[np.min(ind)]
    x = refpixel.x
    top_edge = Pixel(x, topy)
    bottom_edge = Pixel(x, boty)
    return top_edge, bottom_edge

def is_too_close_to_border(edges, borderrange):
    """Return True if edge is too close to border."""
    bordermin, bordermax = borderrange
    for edge in edges:
        if (edge.x <= bordermin) or (edge.x >= bordermax) \
        or (edge.y <= bordermin) or (edge.y >= bordermax):
            return True
    return False

def update_refpixel(refpixel, top_edge, bottom_edge, dx):
    """Update the reference pixel."""
    x = top_edge.x + dx
    y = (top_edge.y + bottom_edge.y)/2.
    return Pixel(x, y)

def is_too_far(edges, refpixel, maxdistance):
    """Return True if the edge is too far from the refpixel."""
    for edge in edges:
        distance = np.sqrt( (edge.x-refpixel.x)**2 + (edge.y-refpixel.y)**2 )
        if distance > maxdistance:
            return True
    return False

def sort_xy(x, y):
    """Sort some point (x,y) for increading x."""
    ind_sorted = np.argsort(x)
    return x[ind_sorted], y[ind_sorted]

def sort_edges(edges):
    """Return sorted edges for x."""
    xedges, yedges = map(np.array, zip(*edges))
    xedges_sorted, yedges_sorted = sort_xy(xedges, yedges)
    edges_sorted = []
    for x, y in zip(xedges_sorted, yedges_sorted):
        edges_sorted.append(Pixel(x=x, y=y))
    return edges_sorted


class SpectralOrder:
    """A spectral order."""
    def __init__(self, name, center):
        self.name = name
        self.center = center
        self.top_edges = []
        self.bottom_edges = []

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.name)

    def append_edges(self, top_edge, bottom_edge):
        """Append new edges to spectral order."""
        self.top_edges.append(top_edge)
        self.bottom_edges.append(bottom_edge)

    def detect_edges(self, img, borderrange=(2,1022), stepsize=5,
                     maxdistance = 15, sigma=7., osf=10):

        startpixel = self.center
        imgsize = img.shape[0]
        y = np.linspace(1, imgsize, imgsize)
        y_oversampled = np.linspace(1, imgsize, imgsize*osf)
        leftside = range(startpixel.x, 1, -stepsize)
        rightside = range(startpixel.x, imgsize, stepsize)
        for side in (leftside, rightside):
            refpixel = startpixel # The reference pixel indicated a good
            # guess of the spectral order center and is updated for each
            # time we change column
            for x in side:
                # Find the y edges
                img_slice = img[:, pixel_to_ind(x)]
                img_slice_oversampled = np.interp(y_oversampled, y, img_slice)
                kernel = gauss_func(x=y_oversampled, mu=imgsize/2., sigma=sigma)
                img_slice_oversampled_smoothend = convolve_with_kernel(
                        img_slice_oversampled, kernel)
                grad = get_grad(img_slice_oversampled_smoothend)
                yedges = y_oversampled[get_argmaxs(grad)]

                # Assume closest two are the right edges
                closest_edges = find_closest(yedges, refpixel)

                # Check if the next edges are not too far away from our guess
                if is_too_far(closest_edges, refpixel, maxdistance) or \
                is_too_close_to_border(closest_edges, borderrange):
                    break
                else:
                    top_edge, bottom_edge = closest_edges
                    self.append_edges(top_edge, bottom_edge)

                    # Update reference pixel for next column
                    dx = (side[1]-side[0])
                    refpixel = update_refpixel(refpixel, top_edge,
                                               bottom_edge, dx)

        # Lastly, sort the edges
        self.top_edges = sort_edges(self.top_edges)
        self.bottom_edges = sort_edges(self.bottom_edges)

    def get_edges_xy(self):
        """Return x,y of the edges."""
        xtop, ytop = map(np.array, zip(*self.top_edges))
        xbot, ybot = map(np.array, zip(*self.bottom_edges))
        return xtop, ytop, xbot, ybot

    def plot_edges_as_overlay(self, **kwargs):
        """Plot the edges. Can be used to overlay on an image."""
        color = kwargs.pop('color', 'r')
        label = kwargs.pop('label', '')
        xtop, ytop, xbot, ybot = self.get_edges_xy()
        plt.scatter(pixel_to_ind(xbot), pixel_to_ind(ybot), color=color,
                    label=label, marker='x', s=100, alpha=0.25)
        plt.scatter(pixel_to_ind(xtop), pixel_to_ind(ytop), color=color,
                    marker='x', s=100, alpha=0.25)

    def fit_traces(self, deg=4):
        """Fit a low-order polynomial tracing the edges."""
        xtop, ytop, xbot, ybot = self.get_edges_xy()
        self.top_coefs = polynomial.polyfit(xtop, ytop, deg)
        self.bot_coefs = polynomial.polyfit(xbot, ybot, deg)

    def get_traces_xy(self):
        """Get the x, y values of a trace."""
        xtop, ytop, xbot, ybot = self.get_edges_xy()
        top_trace = polynomial.polyval(xtop, self.bot_coefs)
        bot_trace = polynomial.polyval(xtop, self.top_coefs)
        return xtop, top_trace, xbot, bot_trace

    def plot_traces_as_overlay(self, **kwargs):
        """Plot the traces. Can be used to overlay on an image."""
        color = kwargs.pop('color', 'r')
        label = kwargs.pop('label', '')
        xtop, ytop, xbot, ybot = self.get_edges_xy()
        yfit_top = polynomial.polyval(xtop, self.top_coefs)
        plt.plot(pixel_to_ind(xtop), pixel_to_ind(yfit_top),
                 label=label, color=color, lw=2)
        yfit_bot = polynomial.polyval(xtop, self.bot_coefs)
        plt.plot(pixel_to_ind(xtop), pixel_to_ind(yfit_bot),
                 color=color, lw=2)

    def plot(self, img):
        """Plot the edges, traces and image below."""
        self.plot_traces_as_overlay()
        self.plot_edges_as_overlay()
        plt.imshow(testimg, vmin=0, vmax=2000, cmap='Greys_r', alpha=0.5)

        xtop, ytop, xbot, ybot = self.get_edges_xy()
        plt.ylim(min([min(ytop),min(ybot)]), max([max(ytop),max(ybot)]))
        plt.show()

    def warp(self, img):
        """Warp the spectral order using the top, bottom trace coefs."""
        xtop, ytop, xbot, ybot = self.get_edges_xy()
        pass


def load_spectral_orders(fpath):
    """Return a list of spectral orders with centers loaded from a .txt file."""
    spectral_orders = []
    centers = np.loadtxt(fpath, skiprows=1, delimiter=',', dtype=int)
    for m, (x, y) in enumerate(centers, 1):
        spectral_order = SpectralOrder(name=m, center=Pixel(x,y))
        spectral_orders.append(spectral_order)

    return spectral_orders

# 
# #%%
# inputdir = '/home/lennart/api/measure/output_pipeline/kelt7/darkcorr/flats_different_elevations'
# flats = get_imgs_in_dir(inputdir)
# elevations = get_elevation_from_fnames(inputdir)
# testimg = flats[0]
#
# #%%
# spectral_orders = []
# #%%
# for flat in (flats[0], flats[1], flats[4], flats[15]):
#     order_centers = load_spectral_orders(fpath='order_centers.txt')
#     spectral_order = order_centers[4]
#     settings = {
#             "borderrange" : (2,1022),
#             "stepsize" : 5,
#             "maxdistance" : 15,
#             "sigma" : 5.,
#             "osf" : 10
#     }
#     spectral_order.detect_edges(img=flat, **settings)
#     spectral_order.fit_traces()
#     spectral_orders.append(spectral_order)
#
# #%%
# for n, (elevation, spectral_order) in enumerate(zip(elevations[[0, 1, 4, 15]], spectral_orders)):
#     spectral_order.plot_edges_as_overlay(color=TABLEAU20[n*2])
#     spectral_order.plot_traces_as_overlay(color=TABLEAU20[n*2], label=elevation)
#
# plt.imshow(testimg, vmin=0, vmax=2000, cmap='Greys_r', alpha=0.5)
# plt.ylim(150, 180)
# plt.xlim(450, 500)
# plt.legend()
# plt.savefig('/home/lennart/api/measure/plots/flats_traces_multiple_elevations.pdf')
# #%%
#
#
# img = testimg
# osf = 100
#
# imgsize = img.shape[0]
# warped_spectral_order = np.zeros(shape=(osf, imgsize))
# x, _, _, _ = spectral_order.get_edges_xy()
# xrange = np.arange(1, imgsize+1)
# yrange = np.arange(1, imgsize+1)
# xrange_warped = np.arange(x.min(), x.max()+1)
# top_trace = polynomial.polyval(xrange, spectral_order.top_coefs)
# bot_trace = polynomial.polyval(xrange, spectral_order.bot_coefs)
# for x, ytop, ybot in zip(xrange_warped, top_trace, bot_trace):
#     img_slice = img[:, pixel_to_ind(x)]
#     yrange_warped = np.linspace(ybot, ytop, osf)
#     warped_spectral_order[:, pixel_to_ind(x)] = np.interp(x=yrange_warped,
#                          xp=xrange, fp=img_slice)
#
# #%%
# plt.imshow(warped_spectral_order)
# #%%
# plt.plot(np.sum(warped_spectral_order, axis=0))
