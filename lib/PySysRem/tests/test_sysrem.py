""" Test the sysrem file."""

from __future__ import print_function, division
import os
import pytest

import numpy as np
import astropy.io.ascii as at
import astropy.io.fits as fits

import source_lc
import sysrem

fake_epochs = np.linspace(0,10,100)
fake_flux = np.sin(fake_epochs)
fake_flags = np.zeros(100,int)
fake_bad = np.array([3,27,52,88])
fake_flags[fake_bad] = 1
true_period = 2*np.pi
check_res = np.median(fake_flux[~fake_bad])
fake_errors = np.ones_like(fake_flux)
fake_else = np.zeros_like(fake_flags)

base_lc = source_lc.source(fake_flux, fake_errors, fake_epochs, fake_flags,"")

def test_matrix_err():
    star_list = []
    for i in range(3):
        star = source_lc.source(fake_flux, fake_errors, fake_epochs, fake_flags,
                                str(i))
        star_list.append(star)

    res, err, meds, slist = sysrem.generate_matrix(star_list)

    check_err = (err==1) | (err>10**19)

    assert np.all(check_err)

def test_res_shape():
    star_list = []
    for i in range(3):
        star = source_lc.source(fake_flux, fake_errors, fake_epochs, fake_flags,
                                str(i))
        star_list.append(star)

    res, err, meds, slist = sysrem.generate_matrix(star_list)

    check_shape = np.shape(res)

    assert check_shape[0]==3
    assert check_shape[1]==100

def test_array_len():
    star_list = []
    for i in range(3):
        star = source_lc.source(fake_flux+i, fake_errors, fake_epochs, fake_flags,
                                "faketest/star_{0}.txt".format(i))
        star_list.append(star)

    res, err, meds, slist = sysrem.generate_matrix(star_list)

    len_orig = len(slist[0].orig_epochs)
    assert len_orig==100

def test_sysrem_fakes():
    star_list = []
    for i in range(3):
        star = source_lc.source(fake_flux+i, fake_errors, fake_epochs, fake_flags,
                                "faketest/star_{0}.txt".format(i))
        star_list.append(star)

    sysrem.sysrem(star_list)

    fake_lc = at.read("faketest/star_1.sysrem.txt")
    lenfake = len(fake_lc)
    # The output light curve should be as long as the entire original lc
    assert lenfake==100

def test_real_ptf_stars():
    filenames = ["lc_00001_calib.txt", "lc_00002_calib.txt",
                 "lc_00003_calib.txt", ]
    star_list = []
    for i in range(3):
        star = source_lc.source.from_ptf(filenames[i])
        star_list.append(star)
    sysrem.sysrem(star_list)

    old_lc = np.loadtxt("lc_00001_sysrem.txt",usecols=np.arange(8))
    old_flux = old_lc[:,1]
    new_lc = at.read("lc_00001_calib.sysrem.txt")
    new_flux = new_lc["mags"]

    flux_diff = np.abs(old_flux - new_flux)
    check_diff = (flux_diff<1e-10) | (new_flux<=-9998)

    assert np.all(check_diff)

data = os.path.join(os.path.dirname(__file__),"k2sff_211889983.fits")
# Extract the original file for testing

def test_real_k2sff_stars():
    # These all have the same epoch arrays
    filenames = ["k2sff_211889983.fits",
                 "k2sff_211892153.fits", "k2sff_211892173.fits"]
    nstars = len(filenames)
    star_list = []
    for i in range(nstars):
        star = source_lc.source.from_k2sff(filenames[i])
        star_list.append(star)
    sysrem.sysrem(star_list)

def test_sysrem_fix_epochs():
    # These have different-length light curves
    filenames = ["k2sff_211887567.fits","k2sff_211889983.fits",
                 "k2sff_211892153.fits", "k2sff_211892173.fits"]
    nstars = len(filenames)
    star_list = []
    ep_lengths = np.zeros(nstars,int)
    for i in range(nstars):
        star = source_lc.source.from_k2sff(filenames[i])
        star_list.append(star)
        ep_lengths[i] = len(star.epochs)
    old_stars = star_list

    if np.all(ep_lengths==ep_lengths[0])==False:
        new_list = source_lc.fix_epochs(star_list)

    sysrem.sysrem(new_list)
