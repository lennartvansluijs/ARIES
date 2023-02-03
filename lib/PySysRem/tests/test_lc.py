""" Test the light curve files."""

from __future__ import print_function, division
import os
import pytest

import numpy as np
import astropy.io.ascii as at
import astropy.io.fits as fits

from .. import source_lc

fake_epochs = np.linspace(0,10,100)
fake_flux = np.sin(fake_epochs)
fake_flags = np.zeros(100,int)
fake_bad = np.array([3,27,52,88])
fake_flags[fake_bad] = 1
true_period = 2*np.pi
fake_errors = np.ones_like(fake_flux)
fake_else = np.zeros_like(fake_flags)

def test_lc():
    lc = source_lc.source(fake_flux, fake_errors, fake_epochs, fake_flags,"")

base_lc = source_lc.source(fake_flux, fake_errors, fake_epochs, fake_flags,"")

def test_clean():
    lc = source_lc.source(fake_flux, fake_errors, fake_epochs, fake_flags,"")
    old_flux = lc.mags
    lc.clean_up()
    assert len(lc.mags)==(len(fake_flux) - len(fake_bad))

def test_median():
    med,std = base_lc._calc_stats(fake_else)
    assert med==0

def test_stdev():
    med,std = base_lc._calc_stats(fake_else)
    assert std==0
