#!/usr/bin/env python
# coding: utf-8

# Pre-processing: cross-talk correction

import argparse
import os
import sys


import numpy as np
from astropy.io import fits

from settings import ARIES_BASE_DIR, DATA_BASE_DIR
sys.path.append(ARIES_BASE_DIR)
from aries.crosstalkcorr import Corquad, CorquadFitter, DEFAULT_CORQUAD_COEFS


# Print to terminal
print('-'*50)
print('1. Cross-talk correction')
print('-'*50)


# Get target info from header arguments
parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdate', type=str)

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate

# Algorithm parameters
SIGMA_HOTS = 5
NMAX_HOTS = 100
DKERNEL = 0.01 # DKERNEL stepsize used in grid search
NGRID = 3 # NGRID x NGRID x NGRID grid search

OBSERVATION_BASE_DIR = os.path.abspath(DATA_BASE_DIR + f'/{targetname}/{obsdate}')
print('Input: 'f'{targetname}/{obsdate} raw data.')
# Estimate crosstalk coeficients
print('\n\tEstimating cross-talk coeficients...')
darks_dirname = os.path.abspath(OBSERVATION_BASE_DIR+'/raw')
output_dirname = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/corquad_fitter')

corquad_path = os.path.abspath(ARIES_BASE_DIR+'/bin/corquad-linux')
corquad_fitter = CorquadFitter(inputdir = darks_dirname,
                               outputdir = output_dirname,
                               executable_path = corquad_path)

kernels = (DEFAULT_CORQUAD_COEFS['kern0'],
           DEFAULT_CORQUAD_COEFS['kern1'],
           DEFAULT_CORQUAD_COEFS['kern2'])
gridrange = [np.linspace(kernel - DKERNEL, kernel + DKERNEL, NGRID) for kernel in kernels]
corquad_fitter.gridrange = gridrange

settings = {
    'silent' : False,
    'save' : True,
    'plot' : True,
    'refpixel' : (680,447), # reference pixel used to estimate best fit
    'dw' : 10 # boxsize used around reference pixel used to estimate best fit
}
bestcoefs = corquad_fitter.run(**settings)
corquad_fitter.cleanup()
print('Done.')


# Run with best fit crosstalk coeficients
outputdir = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/corquad')
inputdir = os.path.abspath(OBSERVATION_BASE_DIR+'/raw')

print('\n\tRunning corquad on all images...')
corquad = Corquad(outputdir = outputdir,
                  coefs = bestcoefs,
                  executable_path=corquad_path)
corquad.run(inputdir)
print('Done.')
