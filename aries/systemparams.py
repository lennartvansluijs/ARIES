import astropy.units as u
import numpy as np

def targetname_to_key(targetname):
    """"""
    t = {
        'kelt7' : 'KELT-7',
        'HD46375_DS' : 'HD 46375',
        'wasp33' : 'wasp33',
        'HD143105' : 'HD 143105',
        'tauBoo' : 'tau Boo A b'
    }
    return t[targetname]


# Nugroho et al. find (Kp, vsys) = (237.5, -1.5)
systems = {
    'KELT-7' : {
        'period' : 2.7347749, # day
        'tprimary' : 2456223.9592, # primary transit time in BJD,
        'mstar' : 1.535, # Msun
        'semimajor' : 0.04415, # au
        'inclination' : 83.76, # degree
        'teff_star' : 6789.0, # kelvin
        'rp' : 1.533, # rjup
        'rstar' : 1.732, # rsun
        'vsys': 40.75e3, # m/s
        'occultation_phases' : None,
        'kp': 174582.713888605 # m/s
    },

    # checked 09-12-2020 http://exoplanet.eu/catalog/wasp-33_b/
    'wasp33' : {
        'period' : 1.21986967, # day
        'tprimary' : 2454590.17936, # primary transit time in BJD,
        'mstar' : 1.495, # Msun
        'semimajor' : 0.02558, # au
        'inclination' : 87.7, # degree
        'teff_star' : 7400, # kelvin
        'rp' : 1.603, # rjup
        'rstar' : 1.444, # rsun 'vsys' : -9.2e3, # m/s, Simbad. metnioned in Serindag et al. (2020) from Nugroho et al. (2017) #'kp' : 237.5e3, #m/s Nugroho et al. (21017), values looked up at Simbad http://simbad.u-strasbg.fr/simbad/sim-id?protocol=html&Ident=WASP-33
        'teff_planet' : 2782.0, # K as on http://exoplanet.eu/catalog/wasp-33_b/ 16/03/2021
        'vsys' : -0.3e3, # m/s, see intro file:///home/lennart/Downloads/Nugroho_2021_ApJL_910_L9.pdf
        'kp' : 230.9e3, #m/s again see paper^^^ Both values are the one derived by Nugroho et al.
        'b' : 0.21, # impact parameter
        "occultation_phases" : (
            0.453595069688914,
            0.4636968695458717,
            0.5363031304541284,
            0.546404930311086
        )  #  Calculated using equations below
    },

    'HD 46375' : {
        'period' : 3.02358, # day
        'tprimary' : None, # primary transit time in BJD,
        'tperi' : 2451920.69867877, #2451920.7 = JD
        'mstar' : 0.91, # Msun
        'mplanet_sini' : 0.23, # Mjup
        'eccentricity' : 0.0524,
        'semimajor' : 0.041, # au
        'inclination' : None, # degree
        'occultation_phases' : None,
        'teff_star' : 5199.0, # kelvin
        'rp' : 1.02, # rjup
        'rstar' : 1.0, # rsun
        'vsys': -0.979e3, # m/s, https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=HD+46375&submit=SIMBAD+search
        'kp': 140.320e3, # (estimated) in m/s
        'omega' : 113.7, # degree +-34.3 http://exoplanet.eu/catalog/hd_46375_a_b/
        'T_equ' : 1514 # Kelvin, from MEASURE excel sheet Jayne
    },

    'HD 143105' : {
        'period' : 2.1974, # day
        'tprimary' : None, # primary transit time in BJD,
        'tperi' : 2456531.344, # in JD
        'mstar' : 1.51, # Msun
        'mplanet_sini' : 1.21, # Mjup
        'eccentricity' : 0.07,
        'semimajor' : 0.0379, # au
        'inclination' : None, # degree
        'occultation_phases' : None,
        'teff_star' : 6380.0, # kelvin
        'rp' : 1.1985, # guessed from mass * sin i using maximal inclination; in rjup
        'rstar' : 1.4, # rsun, from Jayne MEASUE excel sheet - no source
        'vsys': 15.94e3, # m/s, https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=HD+46375&submit=SIMBAD+search
        'kp': 110e3, # guess from Dion's thesis (estimated) in m/s
        'omega' : None, # degree +-34.3 http://exoplanet.eu/catalog/hd_46375_a_b/
        'T_equ' : 2389  # Kelvin, from MEASURE excel sheet Jayne
    },

#     'ID' : {
#         'period' : None, # day
#         'tprimary' : None, # primary transit time in BJD,
#         'mstar' : None, # Msun
#         'semimajor' : None, # au
#         'inclination' : None, # degree
#         'teff_star' : None, # kelvin
#         'rp' : None, # rjup
#         'rstar' : None, # rsun
#         'vsys': None, # m/s
#         'kp': None # m/s
#     }
        'tau Boo A b' : {
            'period' : 3.31249, # day
            'tprimary' : None, # primary transit time in BJD,
            'tperi' : 2450529.2, # in JD
            'mstar' : 1.3, # Msun
            'mplanet_sini' : 4.13, # Mjup
            'eccentricity' : 0.0787,
            'semimajor' : 0.046, # au
            'inclination' : 45.0, # degree
            'occultation_phases' : None,
            'teff_star' : 6309.0, # kelvin
            'rp' : 1.06, # guessed from mass * sin i using maximal inclination; in rjup
            'rstar' : 1.331, # rsun, from Jayne MEASUE excel sheet - no source
            'vsys': -15.4e3, # m/s, Pelletier et al 2021
                            'kp': 108.2e3, # Pelletier et al. 2021
            'omega' : 218.4, # degree +-34.3 http://exoplanet.eu/catalog/hd_46375_a_b/
            'T_equ' : 2139,  # Kelvin, from MEASURE excel sheet Jayne
            'vrot' : None # unknown as far as I am aware
    },
}

def calc_transit_duration(P, Rp, Rs, b, a):
    return (P/(np.pi*u.rad)) * np.arcsin(( np.sqrt( ((Rp+Rs)**2 - (b*Rs)**2)) / a ))

def calc_Ttot(P, Rs, k, b, i, a):
    """Calculate the total transit duration between TI-TIV for a circular orbit."""
    return (P/(np.pi*u.rad)) * np.arcsin((Rs * np.sqrt((1+k)**2 - b**2))/(a * np.sin(i)))

def calc_Tfull(P, Rs, k, b, i, a):
    """Calculate the full transit duration between TII-TIII for a circular orbit."""
    return (P/(np.pi*u.rad)) * np.arcsin((Rs * np.sqrt((1-k)**2 - b**2))/(a * np.sin(i)))

def is_grazing(k, b):
    """Check if a system is grazing or fully eclpised."""
    if 1 - k > b:
        return False
    else:
        return True

def calc_occultation_phases(Ttot, Tfull, P):
    """Return orbital phases at TI-TIV for the occultation. Assumes a circular orbit."""
    phiI = 0.5 - (1./2.)*(Ttot/P).to(u.dimensionless_unscaled).value
    phiII = 0.5 - (1./2.)*(Tfull/P).to(u.dimensionless_unscaled).value
    phiIII = 0.5 + (1./2.)*(Tfull/P).to(u.dimensionless_unscaled).value
    phiIV = 0.5 + (1./2.)*(Ttot/P).to(u.dimensionless_unscaled).value
    return (phiI, phiII, phiIII, phiIV)
