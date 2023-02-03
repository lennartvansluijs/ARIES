import os
import subprocess
import numpy as np
from astropy.io import fits
import astropy.units as u

class cd:
    """Context manager for changing the current working directory."""
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)

def run_with_py27(command, inputdir):
    """Run a command using a Python 2.7 enviorenment."""
    with cd(inputdir):
        subprocess.call(['conda','activate','py27'])
        status = subprocess.run(command.split(), capture_output=True, shell=True)
        subprocess.call(['conda','deactivate'])
    return status

def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

def save_imgs_as_fits(imgs, pathouts):
    """Save list of ndarray images as .fits files at output paths."""
    for img, pathout in zip(imgs, pathouts):
        hdu = fits.PrimaryHDU(img)
        hdul = fits.HDUList([hdu])
        hdul.writeto(pathout+'.fits', overwrite=True)


def phase_filter(phase_obs, occultation_phases, mode='use_all'):
    if mode == 'use_all':
        return np.ones(len(phase_obs), dtype='bool')
    elif mode == 'in_ingress':
        return np.logical_and(phase_obs > occultation_phases[0], phase_obs < occultation_phases[1])
    elif mode == 'in_egress':
        return np.logical_and(phase_obs > occultation_phases[2], phase_obs < occultation_phases[3])
    elif mode == 'in_full_occultation':
        return np.logical_and(phase_obs > occultation_phases[1], phase_obs < occultation_phases[2])
    elif mode == 'out_of_occultation':
        return np.logical_or(phase_obs < occultation_phases[0], phase_obs > occultation_phases[3])
    elif mode == 'pre_occultation':
        return (phase_obs < occultation_phases[0])
    elif mode == 'post_occultation':
        return (phase_obs > occultation_phases[3])
    elif mode == 'symmetric':
        phi_min, phi_max = phase_obs.min(), phase_obs.max()
        delta_phi_symm = min(0.5-phi_min, phi_max-0.5)
        return np.logical_or(
            np.logical_and(phase_obs >= 0.5-delta_phi_symm, phase_obs < occultation_phases[0]),
            np.logical_and(phase_obs > occultation_phases[3], phase_obs <= 0.5+delta_phi_symm)
        )
    elif mode == 'symmetric_post':
        delta_phi_symm = 0.5-0.33
        print(phase_obs[np.logical_and(phase_obs > occultation_phases[3], phase_obs <= 0.5+delta_phi_symm)])
        print(sum(np.logical_and(phase_obs > occultation_phases[3], phase_obs <= 0.5+delta_phi_symm)))
        return np.logical_and(phase_obs > occultation_phases[3], phase_obs <= 0.5+delta_phi_symm)
    else:
        raise ValueError('Phase mode not understood.')


def load_planet_synthetic_spectrum(fpath_model, mode):
    """Load template model wavelength and flux."""
    if mode == 'phoenix_josh':
        data = np.genfromtxt(fpath_model)
        template_wav = data[:,0]/1e4  #  Convert Angstrom to micron
        template_spec = 10**data[:,1]
        return template_wav, template_spec
    elif mode == 'gcm_elsie':
        head = np.loadtxt(fpath_model,max_rows=1)
        Rp = head[2]
        data = np.loadtxt(fpath_model,skiprows=1)
        wl = data[:,0]
        frac = data[:,1]
        Ltot = data[:,2]
        Fp = (frac[:] * Ltot[:]) / (Rp**2) # removed factor pi in the denominator here
        return wl, Fp
    elif mode == 'petitRADTRANS':
        data = np.genfromtxt(fpath_model)
        template_wav = data[:,0] # loaded in micron
        template_spec = data[:,1] * 1e6
        return template_wav, template_spec
    elif mode == 'crires':
        data = np.loadtxt(fpath_model)
        template_wav = data[:,0][::-1] # revert order
        conversion_factor = ((1. * u.W * u.m**-2 * u.cm**-1).cgs).value * np.pi # 1e3 and a conversion for starradian
        template_spec = data[:,1][::-1] * conversion_factor # possibly need to add a conversion here later.
        return template_wav, template_spec
    else:
        raise ValueError('Unkown mode.')


# # Create S/N plot for all individual nights
#     for ndate, obsdate in enumerate(obsdates):
#         # mask phases outside of range
#         selected_phases_combined = phase_filter(
#             phase_obs=phase_combined,
#             occultation_phases=systems[targetname]["occultation_phases"],
#             mode=phase_filter_mode
#         )
#         # only selected phases of some nights
#         is_correct_night = (obsdateid_combined == ndate)
#         selected_phases_combined = np.logical_and(selected_phases_combined, is_correct_night)

#         # Cross correlation matrix for multiple vsys, kp values
#         trial_kp, trial_vsys = (dkp_all + systems[targetname]['kp'], dvsys_all + systems[targetname]['vsys'])
#         snrmatrix, sigmamatrix = calc_detection_matrices2(ccmatrix=ccmatrix_combined[selected_phases_combined,:],
#                                                         dvsys_all=dvsys_all,
#                                                         dkp_all=dkp_all,
#                                                         phase=phase_combined[selected_phases_combined],
#                                                         vbary=vbary_combined[selected_phases_combined],
#                                                         vsys=systems[targetname]['vsys'], # used to shift to planet rest frame
#                                                         rv_sample=rv_sample,
#                                                         kp=systems[targetname]['kp'],
#                                                         radius=TTEST_OUT_OF_TRAIL_RADIUS,
#                                                         trail_width=TTEST_TRAIL_WIDTH)

#         # save T-test and snr matrix
#         fname = os.path.join(dirout, f'snrmatrix_{obsdate}.fits')
#         fits.writeto(fname, snrmatrix, overwrite=True)
#         fname = os.path.join(dirout, f'sigmamatrix_{obsdate}.fits')
#         fits.writeto(fname, sigmamatrix, overwrite=True)
#         fname = os.path.join(dirout, 'grid.txt')
#         header = f"center: (vsys, kp) = ({systems[targetname]['vsys']}, {systems[targetname]['kp']}) \n"
#         "grid: dvsys_all (m/s) | dkp_all (m/s)"
#         np.savetxt(fname, np.c_[dvsys_all, dkp_all], header=header, delimiter=',')
#         center = (systems[targetname]['vsys'], systems[targetname]['kp'])
#         with open(os.path.join(dirout,'center.pickle'), 'wb') as f:
#             pickle.dump(center, f)
