import argparse
import os
import pickle
import sys

import sys
from settings import ARIES_BASE_DIR, DATA_BASE_DIR
CERES_BASE_DIR = os.path.abspath(ARIES_BASE_DIR+'/lib/ceres')
sys.path.append(ARIES_BASE_DIR)
sys.path.append(CERES_BASE_DIR+'/utils/Correlation')
sys.path.append(CERES_BASE_DIR+'/utils/GLOBALutils')
sys.path.append(CERES_BASE_DIR+'/utils/OptExtract')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import GLOBALutils
from GLOBALutils import get_them
from GLOBALutils import retrace
import Marsh

from aries.traces import load_refcoefs
from aries.traces import calc_shift_and_newcoefs
from aries.traces import is_science, is_flat
from aries.traces import plot_img_with_traces
from aries.traces import plot_1d_cut_with_traces


# Print to terminal
print('-'*50)
print('3.a Find Echelle traces solution')
print('-'*50)


# Get target info from header arguments
parser = argparse.ArgumentParser()
parser.add_argument('-targetname', type=str)
parser.add_argument('-obsdate', type=str)

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate

OBSERVATION_BASE_DIR = os.path.abspath(DATA_BASE_DIR + '/{}/{}'.format(targetname, obsdate))


# <b>Algorithm parameters
TRACE_APERTURE = 10
TRACE_DEGREE = 4
MAX_NORDERS = 26
# ---

# Find flat echelle traces



dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr'.format(targetname))
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces'.format(targetname))
if not os.path.exists(dirout):
    os.mkdir(dirout)

fnames = [fname for fname in os.listdir(dirin) if is_flat(fname)]
fnames = np.sort(fnames)

prev_coefs_all = None
for n, fname in enumerate(fnames):
    img_path = os.path.join(dirin, fname)
    img = np.array(fits.getdata(img_path))
    
    print "Tracing Echelle orders for " + img_path + " ({}/{})...".format(n+1, len(fnames))
    print "\tFinding traces..."
    try:
       coefs_all, norders = get_them(sc=img,
                                     exap=TRACE_APERTURE,
                                     ncoef=TRACE_DEGREE,
                                     maxords=MAX_NORDERS)
    except:
        print('No trace solution found by get_them. Setting equal to solution found for the previous frame')
        coefs_all = prev_coefs_all
    finally:
        prev_coefs_all = coefs_all
    print "Done."
    
    output_path = os.path.join(dirout, 'trace_'+fname[:-5]+'.pkl')
    trace_dict = {'coefs_all' : coefs_all,
                  'norders' : norders}
    pickle.dump(trace_dict, open(output_path, 'w'))
    print output_path


# Now finally let's check how well the traces were fitted for each of the images by overplotting the traces and plotting a 1D cut again.
dirin_imgs = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr')
dirin_traces = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces')
dirout = os.path.join(dirin_traces, 'plots')
if not os.path.exists(dirout):
    os.mkdir(dirout)

for fname in fnames:
    img_path = os.path.join(dirin_imgs, fname)
    img = np.array(fits.getdata(img_path))

    input_path = os.path.join(dirin_traces, 'trace_'+fname[:-5]+'.pkl')
    trace_dict = pickle.load(open(input_path))
    coefs_all = trace_dict['coefs_all']
    
    plot_img_with_traces(img, coefs_all)
    plot_output_path = os.path.join(dirout, 'plot_img_traces_'+fname[:-5])
    plt.savefig(plot_output_path+'.png', dpi=300)
    plt.close()
    
    plot_1d_cut_with_traces(img, coefs_all)
    plot_output_path = os.path.join(dirout, 'plot_img1dcut_traces_'+fname[:-5])
    plt.savefig(plot_output_path+'.png', dpi=300)
    plt.close()


# Find science echelle traces
#
# The science frames are more difficult to fit due to the discontiunity of the spectra and detector artifacts. Additionaly, the orders move around due to instrument flexure. CERES contains the function ```retrace()``` to correct for a shift of the traces in the science frames. This function cross-correlates the flat traces with the science frames to determine the shift and updates the trace coeficients accordingly.

dirin_imgs = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr/')
dirin_traces = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces/')
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces/retrace')
if not os.path.exists(dirout):
    os.mkdir(dirout)

fnames_science = [fname for fname in os.listdir(dirin_imgs) if is_science(fname)]
fnames_flat = [fname for fname in os.listdir(dirin_imgs) if is_flat(fname)]

print "\tLoading all flat trace coefs."
refcoefs_all_flats = load_refcoefs(dirin_traces, fnames_flat)
print "Done."


for n, science in enumerate(fnames_science):
    img_path = os.path.join(dirin_imgs, science)
    img = np.array(fits.getdata(img_path))
    
    print "\tRetracing " + science + " ({}/{})...".format(n+1, len(fnames_science))
    shifts, new_coefs_all_flats = calc_shift_and_newcoefs(img, refcoefs_all_flats)
    print "Done."
    
    # save the updated shifts and trace coeficients
    retrace_dict = {
        'flats' : fnames_flat,
        'shifts' : shifts,
        'new_coefs_all_flats' : new_coefs_all_flats
    }
    output_path = os.path.join(dirout, 'retrace_'+science[:-5]+'.pkl')
    pickle.dump(retrace_dict, open(output_path, 'w'))


# Now we use the trace with the lowest shift as the best trace.


dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces/retrace')
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces')

files = [f for f in os.listdir(dirin) if f.endswith('.pkl')]

FNAME_INDS = slice(8, 21+len(targetname), 1)
for f in files:
    fname = f[FNAME_INDS]
    fpath = os.path.join(dirin, f)
    retrace_dict = pickle.load(open(fpath))
    fnames_flat = np.array(retrace_dict["flats"])
    shifts = np.array(retrace_dict["shifts"])
    new_coefs_all_flats = np.array(retrace_dict["new_coefs_all_flats"])
    
    # find best flat retrace to use
    ind_sorted = np.argsort(shifts)
    coefs_all = new_coefs_all_flats[ind_sorted][0]
    shift = shifts[ind_sorted][0]
    refflat = fnames_flat[ind_sorted][0]
    
    # save science coeficients
    trace_dict = {'coefs_all' : coefs_all,
                  'shift' : shift,
                  'refflat' : refflat}
    output_path = os.path.join(dirout, 'trace_'+fname+'.pkl')
    pickle.dump(trace_dict, open(output_path, 'w'))


# Now let's also create plots for the science traces.
dirin_imgs = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/darkcorr'.format(targetname))
dirin_traces = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces'.format(targetname))
dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/processed/traces/plots'.format(targetname))
if not os.path.exists(dirout):
    os.mkdir(dirout)

fnames = [fname for fname in os.listdir(dirin_imgs) if is_science(fname)]

for fname in fnames:
    img_path = os.path.join(dirin_imgs, fname)
    img = np.array(fits.getdata(img_path))

    input_path = os.path.join(dirin_traces, 'trace_'+fname[:-5]+'.pkl')
    trace_dict = pickle.load(open(input_path))
    coefs_all = trace_dict['coefs_all']
    
    plot_img_with_traces(img, coefs_all)
    plot_output_path = os.path.join(dirout, 'plot_img_traces_'+fname[:-5])
    plt.savefig(plot_output_path+'.png', dpi=300)
    plt.close()
    
    plot_1d_cut_with_traces(img, coefs_all)
    plot_output_path = os.path.join(dirout, 'plot_img1dcut_traces_'+fname[:-5])
    plt.savefig(plot_output_path+'.png', dpi=300)
    plt.close()

