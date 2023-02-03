import argparse
import os
import pickle
import sys
print sys.path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

base = os.path.abspath("../lib/ceres")+'/'
sys.path.append(base+"utils/Correlation")
sys.path.append(base+"utils/GLOBALutils")
sys.path.append(base+"utils/OptExtract")
print sys.path
import Marsh
import GLOBALutils

# Recive input parameters
parser = argparse.ArgumentParser()
parser.add_argument('img_path')
parser.add_argument('-dirout')
parser.add_argument('-ext_aperture')
parser.add_argument('-trace_degree')

# Parse
args = parser.parse_args()
img_path = args.img_path
dirout = args.dirout
ext_aperture = int(args.ext_aperture)
trace_degree = int(args.trace_degree)

img = np.array(fits.getdata(img_path))
fname = os.path.basename(img_path)

print "\tTracing echelle orders..."
coefs_all, norders = GLOBALutils.get_them(sc=img, exap=ext_aperture, ncoef=trace_degree)

# Save trace coefs for all orders
trace_dict = {'coefs_all' : coefs_all,
              'norders' : norders}
output_path = os.path.join(dirout, 'trace_'+fname[:-5]+'.pkl')
pickle.dump(trace_dict, open(output_path, 'w'))
print "Done."
