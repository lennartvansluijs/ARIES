import matplotlib
matplotlib.use("Agg")
import numpy as np
import scipy
from scipy import signal,interpolate
import scipy.special as sp
import copy
import glob
import os
import sys
from pylab import *
sys.path.append("../utils/GLOBALutils")
import GLOBALutils
from astropy.io import fits as pyfits
from photometry import *

import statsmodels.api as sm
lowess = sm.nonparametric.lowess


def get_bad_files(fpath):
    """
    Get bad files from txt file.
    """
    bad_files = []
    if os.access(fpath, os.F_OK):
        bf = open(fpath)
        linesbf = bf.readlines()
        for line in linesbf:
            bad_files.append(diri+line[:-1])
        bf.close()
    return bad_files

def get_ftype(fname):
    """
    Return type of calibration file based on the file name.
    """
    ftypes = ('science', 'flat', 'dark')
    for ftype in ftypes:
        if ftype in fname:
            return ftype

def FileClassify(diri, log):
    """
    Classifies all files in a directory and writes a night log of science images.
    """
    # define output lists
    sim_sci          = []
    sim_sci_exptimes = []
    flats            = []
    flats_exptimes   = []
    darks            = []
    darks_exptimes   = []

    # open log file
    f = open(log,'w')

    # do not consider the images specified in dir+badfiles.txt
    all_files = glob.glob(diri+"/*fits")
    bad_files = get_bad_files(fpath=os.path.join(diri+'bad_files.txt'))
    good_files = [file for file in all_files if f not in bad_files]

    for fname in good_files:
        ftype = get_ftype(fname)
        hd = pyfits.getheader(fname)
        img = pyfits.getdata(fname)

        # Extract the following keywords
        print hd
        obname = hd['OBJECT']
        ra     = hd['RA']
        delta  = hd['DEC']
        airmass= hd['AIRMASS']
        texp   = hd['EXPTIME']
        uts    = hd['UTSTART']

        # write line to log file
        line = "%-15s %10s %10s %8.2f %4.2f %s %s\n" \
        % (obname, ra, delta, float(texp), float(airmass), uts, fname)
        f.write(line)

        if ftype is 'science':
            sim_sci.append(fname)
            sim_sci_exptimes.append(texp)
        if ftype is 'dark':
            darks.append(fname)
            darks_exptimes.append(texp)
        if type is 'flat':
            flats.append(fname)
            flats_exptimes.append(texp)

    sim_sci = np.array(sim_sci)
    darks = np.array(darks)
    flats = np.array(flats)
    darks_exptimes = np.array(darks_exptimes)
    flats_exptimes = np.array(flats_exptimes)
    sim_sci_exptimes = np.array(sim_sci_exptimes)

    return sim_sci, sim_sci_exptimes, flats, flats_exptimes, darks, darks_exptimes

def MedianCombine(ImgList):
    """
    Median combine a list of images
    """

    if len(ImgList) == 0:
        raise ValueError("empty list provided!")

    imgs = [fits.getdata(img) for img in ImgList]
    median_img = np.median(imgs, axis=2)

    return median_img

def get_master_darks(darks, darks_exptime):
    """Return the master dark in counts/second."""
    darks = np.array([fits.getdata(d) for d in darks])

    SHORT_EXPTIME = np.where(darks_exptime == 60.0)
    LONG_EXPTIME = np.where(darks_exptime == 300.0)

    master_dark_long_exptime = get_median(darks[LONG_EXPTIME])
    master_dark_short_exptime = get_median(darks[SHORT_EXPTIME])

    hots_long_exptime = find_hots(img=master_dark_long_exptime, sigma=5.)
    hots_short_exptime = find_hots(img=master_dark_short_exptime, sigma=5.)
    darks_long_exptime = correct_hots(imgs=darks[LONG_EXPTIME], hots=hots_long_exptime)
    darks_short_exptime = correct_hots(imgs=darks[SHORT_EXPTIME], hots=hots_short_exptime)

    master_dark_long_exptime = get_median(darks_long_exptime)
    master_dark_short_exptime = get_median(darks_short_exptime)

    return master_dark_long_exptime, master_dark_short_exptime
