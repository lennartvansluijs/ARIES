import os
import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import binned_statistic
from scipy import polyval
from aries.preprocessing import robust_polyfit
from telfit import Modeler
from scipy.signal import medfilt
import inspect

print(inspect.getfile(Modeler))
KERNELSIZE = 1e3

def convert_airmass_to_angle(X):
    return np.arccos(1./X) * (180/np.pi) # in degree

def logL_zucker(data, model):
    N = float(len(data))
    cc = np.corrcoef(data, model)[0,1]
    return -0.5 * N * np.log(1-cc**2)

def arange_at_fixed_R(wavmin, wavmax, R):
    """Arange with steps of lambda/R with R the spectral resolution."""
    w = []
    w.append(wavmin)
    while w[-1] < wavmax:
        w.append(w[-1]*(1. + 1./R))
    return np.array(w)

def build_telluric_lib(zenithangle_values,
                       humidity_values,
                       observatory_params,
                       wavgrid_os,
                       dirout,
                       do_plot=True):
    """Build a library of telluric models for a given set of zenith angles and humidities."""
    if not os.path.exists(dirout):
        os.mkdir(dirout)

    # save array of model grid values
    np.save(os.path.join(dirout, 'zenithangle_values.npy'), zenithangle_values)
    np.save(os.path.join(dirout, 'humidity_values.npy'), humidity_values)
    np.save(os.path.join(dirout, 'wavgrid_os.npy'), wavgrid_os)
    
    modeler = Modeler()
    ntotal = len(zenithangle_values) * len(humidity_values)
    niter = 1
    for theta in zenithangle_values:
        for RH in humidity_values:
            modelname = f'telfitmodel_zenithangle-{theta:.0f}_humidity-{RH:.1f}'
            print(f'Model: {modelname} ({niter}/{ntotal})')
            model = modeler.MakeModel(wavegrid=wavgrid_os,
                                      humidity=RH,
                                      lowfreq=1e7/wavgrid_os[-1],
                                      highfreq=1e7/wavgrid_os[0],
                                      angle=theta,
                                      lat=observatory_params['lat'],
                                      alt=observatory_params['alt'], do_rebin=True, vac2air=False)
            
            if do_plot:
                dirout_plots = os.path.join(dirout, 'plots')
                if not os.path.exists(dirout_plots):
                    os.mkdir(dirout_plots)
                
                plt.figure(figsize=(5*1.168, 5))
                plt.plot(model.x/1e3, model.y)
                plt.xlabel('Wavelength (micron)', size=12)
                plt.ylabel('Transmission')
                plt.title(modelname, size=12)
                
                fpathout = os.path.join(dirout_plots, modelname+'.png')
                plt.savefig(fpathout, dpi=150)
                plt.close()
            
            # save each model as a numpy .txt file
            fpathout = os.path.join(dirout, modelname+'.npy')
            data = np.array([model.x, model.y])
            np.save(fpathout, data)
            niter += 1

def load_telluric_models(dirin):
    """Interpolate telluric models from Telfit models for any humidity/angle value."""
    zenithangle_values = np.load(os.path.join(dirin, 'zenithangle_values.npy'))
    humidity_values = np.load(os.path.join(dirin, 'humidity_values.npy'))
    wavgrid_os = np.load(os.path.join(dirin, 'wavgrid_os.npy'))
    
    nx, ny, nz = len(zenithangle_values), len(humidity_values), len(wavgrid_os)
    telluric_models = np.zeros(shape=(nx, ny, nz))
    ntotal = len(zenithangle_values) * len(humidity_values)
    niter = 1
    for i, theta in enumerate(zenithangle_values):
        for j, RH in enumerate(humidity_values):
            modelname = f'telfitmodel_zenithangle-{theta:.0f}_humidity-{RH:.1f}'
            
            # load model data
            fpath = os.path.join(dirin, modelname+'.npy')
            data = np.load(fpath)
            telluric_models[i, j, :] = data[1]
            
    return (zenithangle_values, humidity_values, wavgrid_os), telluric_models

def gaussian_convolution_kernel(wav, resolution, sigma_kernel_cutoff=9, kernelsize_min=11):
    """Returns a Gaussian kernel at the appropriate spectral resolution and size given the spectral sampling rate."""
    # Calculate sampling rate and size of resolution element
    wavmin, wavmax = wav[0], wav[-1]
    npoints = len(wav)
    lambda0 = (wavmax+wavmin)/2.
    delta_lambda = lambda0/resolution
    samplingrate = (wavmax-wavmin)/npoints
    
    # Define kernelsize
    fwhm = delta_lambda/samplingrate
    GAUSSIAN_FWHM_TO_SIGMA = 1./np.sqrt(8*np.log(2))
    sigma_kernel = GAUSSIAN_FWHM_TO_SIGMA * fwhm
    kernelsize = int(sigma_kernel * sigma_kernel_cutoff)
    
    # Ensure odd kernelsize to make kernel symmetric
    if not kernelsize % 2 == 1:
        kernelsize += 1

    # Check if kernelsize is not too small or too large
    if kernelsize > npoints:
        warnings.warn('\nKernelsize larger than total wavelength range.' \
                      '\nSetting kernelsize equal to total number of points.' \
                      '\nConvolution may be incorrect, choose a higher resolution' \
                      'to resolve this.')
    elif kernelsize < kernelsize_min:
        #         warnings.warn(f'\nKernelsize is smaller than minimal kernelsize points using' \
        #                       f' this sampling rate.\nSetting kernelsize equal to {kernelsize_min}.' \
        #                       '\nConvolution may be incorrect, choose a smaller resolution' \
        #                       'to resolve this.')
        kernelsize = kernelsize_min

    kernel = np.exp(-(np.arange(kernelsize)-int(np.floor(kernelsize/2.)))**2/(2.*sigma_kernel**2))
    return kernel/kernel.sum()

def rebin_spectrum(wavegrid, wavegrid_new, spec):
    """Return spectrum rebinned to new wavegrid."""
    # Calculate bin edges
    steps = np.diff(wavegrid_new)
    bin_edges = [wavegrid_new[0] - steps[0]/2.]
    bin_edges += [wavegrid_new[i+1] - steps[i]/2. for i in range(len(wavegrid_new)-1)]
    bin_edges += [wavegrid_new[-1] + steps[-1]/2.]
    
    # Rebin spectrum
    spec_rebinned, _, _ = binned_statistic(x=wavegrid, values=spec, statistic='mean', bins=bin_edges)
    
    # Correct potential negative fluxes
    #spec_rebinned[(spec_rebinned < 0)] = 0.
    
    return spec_rebinned

# def TelluricInterpolator(dirin):
#     """Evaluate tellruic model for given parameters."""
#     (zenithangle_values, humidity_values, wavgrid_os), telluric_models = load_telluric_models(dirin)
#     get_resolution = lambda w: np.mean(w[:-1]/np.diff(w))
#     R_os = get_resolution(wavgrid_os)
#     print(R_os)
#     interp_telluric = RegularGridInterpolator((zenithangle_values, humidity_values, wavgrid_os),
#                                               telluric_models, bounds_error=False, method='linear')
#     def wrapped(wav, zenithangle, humidity, R):
#         if R > R_os:
#             raise ValueError('Resolution should be <= resolution of the model.')
#         kernel_FWHM = R_os/R
#         kernel_std = kernel_FWHM / 2.35482004503
#         kernel = gaussian_kernel(np.arange(len(wavgrid_os)), mu=len(wavgrid_os)/2., sigma=kernel_std)
#         pts = [(zenithangle, humidity, w) for w in wavgrid_os]
#         telluric_Ros = interp_telluric(pts) # linearly interpolate from telluric library for unknown angle/humidity
#         telluric_R = np.convolve(telluric_Ros, kernel, mode='same') # convolve to new spectral resolution
#         telluric_model = np.interp(x=wav, xp=wavgrid_os, fp=telluric_R) # interpolate to desired wavelength grid
#         return wav, telluric_model
#     return wrapped

def TelluricInterpolator(dirin):
    """Evaluate tellruic model for given parameters."""
    (zenithangle_values, humidity_values, wavgrid_os), telluric_models = load_telluric_models(dirin)
    interp_telluric = RegularGridInterpolator((zenithangle_values, humidity_values, wavgrid_os),
                                              telluric_models, bounds_error=False, method='linear')
    def wrapped(wav, zenithangle, humidity, R):
        # interpolate from telluric library
        pts = [(zenithangle, humidity, w) for w in wavgrid_os]
        telluric_model_os = interp_telluric(pts)
        
        # convolve to new spectral resolution R
        convolution_kernel = gaussian_convolution_kernel(wav=wavgrid_os, resolution=R)
        telluric_model_os_convolved = np.convolve(telluric_model_os, convolution_kernel, mode='same') # convolve to new spectral resolution
        
        # Rebin to data wavelength grid
        telluric_model = rebin_spectrum(wavegrid=wavgrid_os, wavegrid_new=wav, spec=telluric_model_os_convolved)
        return wav, telluric_model
    return wrapped

def convert_RH_to_PWV(RH, Tsurf):
    """Convert Relative Humidity to a Percipiral Water Volume"""    
    R_a = 287.058 # ...
    rho_a = 1.255 # ...
    rho_w = 997 # density of water cgs
    g = 9.81 # gravitational acceleration Earth
    
    def RH_to_pwv_integral(T, RH):
        # esat from : https://www.engineeringtoolbox.com/water-vapor-saturation-pressure-air-d_689.html
        esat = np.exp(77.3450 + 0.0057*T - 7235 / T) / T**8.2 # saturation pressre of water
        e = esat * RH/100
        return 10**3 * R_a * rho_a / (rho_w * g ) * 0.622 * e / (R_a*rho_a*T - e)

    pwv, error = quad(RH_to_pwv_integral, 0, Tsurf, args=(RH)) # assumes space as has a zero temperature and we itnegrate until we reach our surface temperature
    return pwv

def locate_2d_max(img, x, y):
    yy, xx = np.meshgrid(x, y)
    imax = np.unravel_index(img.argmax(), img.shape)
    ymax, xmax = xx[imax], yy[imax]
    valuemax = img[imax]
    return xmax, ymax, valuemax

def correct_continuum(wav, spec, nbins=10, polydeg=3, do_plot=True, do_medfilt=True):    
    if do_medfilt:
        spec = medfilt(spec, kernel_size=5) # just to get rid of any outliers
    
    ycont, bin_edges, _ = binned_statistic(wav, spec, bins=nbins, statistic='max')
    xcont, bin_edges, _ = binned_statistic(wav, spec, bins=nbins, statistic=np.argmax)
    xcont = bin_edges[:-1] + np.diff(bin_edges)/2.
    _, coefs, _ = robust_polyfit(x=xcont, y=ycont, deg=polydeg, return_full=True)
    continuum = polyval(coefs, wav)
    
    # correction
    spec_corr = spec / continuum
    
    if do_plot:
        fig, axes = plt.subplots(2,1, figsize=(10,10))
        axes[0].set_title('Before continuum correction')
        axes[0].plot(wav, continuum, color='r')
        axes[0].scatter(xcont, ycont, color='r')
        axes[0].plot(wav, spec, lw=1, color='k')

        axes[1].plot(wav, spec_corr, color='k')
        axes[1].set_title('After continuum correction')
    
    return spec_corr

def VaporPressure(T):
    """
      This function uses equations and constants from
      http://www.vaisala.com/Vaisala%20Documents/Application%20notes/Humidity_Conversion_Formulas_B210973EN-F.pdf
      to determine the vapor pressure at the given temperature

      T must be a float with the temperature (or dew point) in Kelvin
    """
    #Constants
    c1, c2, c3, c4, c5, c6 = -7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.8022502
    a0, a1 = -13.928169, 34.707823
    if T > 273.15:
        theta = 1.0 - T / 647.096
        Pw = 2.2064e5 * np.exp(1.0 / (1.0 - theta) * ( c1 * theta +
                                                       c2 * theta ** 1.5 +
                                                       c3 * theta ** 3 +
                                                       c4 * theta ** 3.5 +
                                                       c5 * theta ** 4 +
                                                       c6 * theta ** 7.5 ))
    elif T > 173.15:
        theta = T / 273.16
        Pw = 6.11657 * np.exp(a0 * (1.0 - theta ** -1.5) +
                              a1 * (1.0 - theta ** -1.25))
    else:
        Pw = 0.0

    return Pw


def humidity_to_ppmv(RH, T, P):
    """
    Given the relative humidity, temperature, and pressure, return the ppmv water concentration
    """
    Psat = VaporPressure(T)
    Pw = Psat * RH / 100.0
    h2o = Pw / (P - Pw) * 1e6
    return h2o

def ppmv_to_humidity(h2o, T, P):
    """
    Given the ppmv water concentration, temperature, and pressure, return the relative humidity
    """
    Psat = VaporPressure(T)
    RH = 100.0*h2o*P/(Psat*(1e6+h2o))
    return RH
