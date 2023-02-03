

# <b> Modules and packages

# In[1]:


import csv
import os
import sys
from settings import ARIES_BASE_DIR, DATA_BASE_DIR
sys.path.append(ARIES_BASE_DIR)
sys.path.append(os.path.abspath(ARIES_BASE_DIR)+'/lib')

from astropy.time import Time
from astropy.io import fits
from astropy.constants import G, au, M_sun, R_sun, R_jup

# from astropy.utils import iers
# iers_a_file = 'my/path/to/finals2000A.all'
# iers.IERS.iers_table = iers.IERS_A.open(iers_a_file)
import barycorrpy
from barycorrpy import utc_tdb, get_BC_vel, get_stellar_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from aries.constants import TABLEAU20
from aries.cleanspec import SpectralCube, SpectralOrder, clip_mask, pca_detrending
from aries.crosscorrelate import get_rvplanet
from aries.preprocessing import get_keyword_from_headers_in_dir, get_fits_fnames
from aries.crosscorrelate import calc_orbital_phase
from aries.crosscorrelate import estimate_kp
from aries.constants import ARIES_NORDERS
from aries.cleanspec import apply_highpass_filter, sliding_window_iter
import matplotlib.gridspec as gridspec
from aries.phasecalc import calc_phase_ecc

from aries.orbit import SOLAR_MASS, JUPITER_MASS, AU
from aries.orbit import Orbit
from aries.orbit import degree_to_radian


# <b> Target specific

# In[2]:


from aries.systemparams import systems, targetname_to_key

# ---
# Parse input parameters
# ---


parser = argparse.ArgumentParser()
parser.add_argument('-targetname')
parser.add_argument('-obsdate')

args = parser.parse_args()
targetname = args.targetname
obsdate = args.obsdate

targetid = targetname_to_key(targetname)
#obsdate = '20151102'

systemparams = systems[targetid]
OBSERVATION_BASE_DIR = os.path.abspath(DATA_BASE_DIR + f'/{targetname}/{obsdate}')
obsname = 'Multiple Mirror Telescope'


# Assign a unique directory name or overwrite an existing directory.

# <b> Constants

# In[3]:


DAY = 24*60*60 # s


# ## Barycentric time and velocity corrections

# In[4]:


dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/raw')
fnames = np.array(get_fits_fnames(dirin, key='science'))
times_utstart = get_keyword_from_headers_in_dir(keyword='UTSTART', dirname=dirin, key='science')
exptimes = get_keyword_from_headers_in_dir(keyword='EXPTIME', dirname=dirin, key='science')
airmass_values = get_keyword_from_headers_in_dir(keyword='AIRMASS', dirname=dirin, key='science')

# conversion of times to JD and BJD
times_jdutc = np.array([Time(t, scale='utc').jd for t in times_utstart]) + (exptimes/2.) / DAY
times_bjdtdb = utc_tdb.JDUTC_to_BJDTDB(times_jdutc, obsname='Multiple Mirror Telescope')[0]
TIMES_SORTED = np.argsort(times_bjdtdb)

dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/meta')
fpathout = os.path.join(dirout, 'science_times.txt')
header = 'dirin: {}; fnames | time [UT start] | exptimes [s] |'          ' time [JDUTC] (mid exposure) | time [BJDTDB] (mid exposure)'.format(dirout)
np.savetxt(fpathout,
           np.c_[fnames[TIMES_SORTED],
                 times_utstart[TIMES_SORTED],
                 exptimes[TIMES_SORTED],
                 times_jdutc[TIMES_SORTED],
                 times_bjdtdb[TIMES_SORTED]],
           fmt='%s', header=header, delimiter=',')


# also save airmass for later usage
fpathout = os.path.join(dirout,'airmass.txt')
np.savetxt(fpathout, airmass_values[TIMES_SORTED])


# Barycentric velocity calculation.

# In[5]:


from barycorrpy.utils import get_stellar_data
stellar_data = get_stellar_data(targetid.upper())[0]
stellar_data['rv'] = systems[targetid]['vsys'] # overwrite the system velocity queried by Simbad
stellar_data.pop('rv')

dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/meta')
fpath = os.path.join(dirin, 'science_times.txt')
fnames = np.loadtxt(fpath, dtype='str', delimiter=',').T[0].astype('str')
times_jdutc = np.loadtxt(fpath, dtype='str', delimiter=',').T[3].astype('float')
vbary = get_BC_vel(JDUTC=times_jdutc, obsname=obsname, **stellar_data)[0]


# ## Calculate target observed planet RV

# Now let's calculate orbital phase. This a circular orbit in other words it assumes an eccentricity of e = 0.

# In[6]:


''' Using Newton's method to find the root of Kepler's equation.
meanAn = mean anomaly (basically 2*pi*phase)
ecc = orbital eccentricity
eps = tolerance to decide when the algorithm has converged
RETURNS the eccentric anomaly E
 '''
def ecc_anomaly(meanAn,ecc,eps=1E-10):
    E0 = meanAn
    diff = 1.0
    while diff > eps:
        E1 = E0 - (E0 - ecc*np.sin(E0)-meanAn)/(1.0-ecc*np.cos(E0))
        diff = np.abs(E1-E0).max()
        E0 = E1.copy()
    return E1

''' Getting the radial velocity vector given the eccentricity 'ecc', argument of
periastron 'omega', phase vector 'ph', systemic velocity 'vsys', orbital RV semi-
amplitude 'kp', and vector of barycentric velocities 'vbary'. Note that vbary is
in the telluric frame and thus has the +sign. You might need to switch to a -sign
if you still use the vbary as extracted from the file headers. '''
def get_eccentric_RV(omega, ecc, ph, vsys, kp, vbary):
    #omega = np.arctan2(hh,kk)
    #ecc = hh**2 + kk**2
    los_peri = 2 * np.pi * ph
    M = los_peri - np.pi/2 - omega
    E = ecc_anomaly(M, ecc)
    f = 2*np.arctan(np.sqrt(1+ecc)/np.sqrt(1-ecc)*np.tan(E/2.0))
    RV = vbary + vsys + kp * (np.cos(f + omega) + ecc*np.cos(omega))
    return RV


# In[7]:


help(calc_phase_ecc)


# In[10]:


dirin = os.path.abspath(OBSERVATION_BASE_DIR+'/meta')
fpath = os.path.join(dirin, 'science_times.txt')
times_bjdtbd = np.loadtxt(fpath, dtype='str', delimiter=',').T[4].astype('float')

#systems[targetid]['kp'], systems[targetid]['vsys'] = 237.5e3, -1.5e3 # change to Nugroho et al. find (Kp, vsys) = (237.5, -1.5)
if systems[targetid]['tprimary'] is not None:
    phase = calc_phase_ecc(times=times_bjdtdb[TIMES_SORTED], Tp=systems[targetid]['tprimary'],
                           Torb=systems[targetid]['period'], e=systems[targetid]['eccentricity'])
else:
    phase = calc_phase_ecc(times=times_bjdtdb[TIMES_SORTED], Tp=systems[targetid]['tperi'],
                           Torb=systems[targetid]['period'], e=systems[targetid]['eccentricity'])
nobs = len(phase)
print('Phases: {}'.format(phase))
# systems[targetid]['kp'] =  estimate_kp(systems[targetid]['semimajor']*au.value, systems[targetid]['mstar']*M_sun.value, systems[targetid]['inclination']*(np.pi/180.))

# stellar_data = get_stellar_data(targetname.upper())
# systems[targetid]['vsys'] = stellar_data[0]['rv']

if systems[targetid]['eccentricity'] == 0.:
    print('Solving for a circular orbit...')
    rvplanet = get_rvplanet(vbary, systems[targetid]['vsys'], systems[targetid
]['kp'], phase=phase)
    rvplanet_os = get_rvplanet(vbary, systems[targetid]['vsys'], systems[targetid]['kp'], phase=np.linspace(0,1,nobs)) # oversampled light curve
elif 0. <= systems[targetid]['eccentricity'] <= 1.:
    print('\tSolving for an eccentric orbit...')
    system = systems['HD 46375']
    orbit_params = {
        't0' : system['tperi'] * DAY,
        'ecc' : system['eccentricity'], # unknown
        'period' : system['period'] * DAY,
        'mass1' : system['mstar'] * SOLAR_MASS,
        'mass2' : system['mplanet_sini'] * JUPITER_MASS,
        'semi_major_axis' : system['semimajor'] * AU,
        'omega' : degree_to_radian(system['omega'])
    }
    orbit = Orbit('HD_46375', **orbit_params)
    orbit.evolve(time=times_bjdtbd*DAY, mode='time')
    rvplanet = orbit.history[3] + systems[targetid]['vsys'] + vbary
    rvplanet_v2 = get_eccentric_RV(
        omega=orbit_params['omega'],
        ecc=orbit_params['ecc'],
        ph=phase,
        vsys=systems[targetid]['vsys'],
        kp=systems[targetid]['kp'],
        vbary=vbary
    )
    rvplanet_os_v2 = get_eccentric_RV(
        omega=orbit_params['omega'],
        ecc=orbit_params['ecc'],
        ph=np.linspace(0,1,100),
        vsys=systems[targetid]['vsys'],
        kp=systems[targetid]['kp'],
        vbary=0.
    )
    orbit = Orbit('HD_46375', **orbit_params)
    orbit.evolve(time=np.linspace(orbit_params['t0'],orbit.period+orbit_params['t0'], 500), mode='time')
    rvplanet_os = orbit.history[3] + systems[targetid]['vsys']

    rvplanet_os_circ = get_rvplanet(0, systems[targetid]['vsys'], systems[targetid]['kp'], phase=np.linspace(0,1,nobs)) # oversampled light curve
print('Done.')

# write to csv file
fname = f'{targetname}_targetinfo.csv'
fpath = os.path.join(dirin, fname)
with open(fpath, "w") as csvfile:
    w = csv.writer(csvfile)
    for key, val in systemparams.items():
        w.writerow([key, val])

dirout = os.path.abspath(OBSERVATION_BASE_DIR+'/meta')
fpathout = os.path.join(dirout, 'vbarycorr.txt')
header = 'phase | vbary (m/s) | planet RV (m/s)'.format(dirout)
print('\tOverwriting exisiting phase, RV solution...')
np.savetxt(fpathout,
           np.c_[[phase, vbary, rvplanet]], header=header)
print('Done.')

phase, vbary, rvplanet = np.loadtxt(fpathout)

plt.figure(figsize=(5*1, 5*1.68))
#plt.plot(rvplanet_os/1e3, np.linspace(0,1,rvplanet_os.size), color='k', label='eccentric')
#plt.scatter( (rvplanet-vbary)/1e3, phase, s=100, facecolors='none', edgecolors=TABLEAU20[0])
plt.scatter( (rvplanet_v2-vbary)/1e3, phase, s=100, facecolors='none', edgecolors=TABLEAU20[6])
plt.plot( (rvplanet_os_v2)/1e3, np.linspace(0,1,rvplanet_os_v2.size), label='eccentric', color=TABLEAU20[6])
plt.plot(rvplanet_os_circ/1e3, np.linspace(0,1,rvplanet_os_circ.size), color='k', ls='--', label='circular')
plt.ylim(0,1)
plt.legend()
plt.ylabel('Orbital phase', size=12)
plt.xlabel('Velocity (km/s)', size=12)
plt.title(f"Phase coverage + planet RV (before barycorr) ({targetname.upper()})", size=12)
plt.close()


# In[ ]:





# In[ ]:
