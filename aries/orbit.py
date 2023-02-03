import sys

from functools import partial
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.animation as animation

ARIES_BASE_DIR = '../..'
sys.path.append(ARIES_BASE_DIR)
from aries.systemparams import systems
from aries.constants import TABLEAU20


class Unit(float):
    """Class to create physical units.

    Units are simply floats, with a simple twist: the string
    representation is changed to a name, set by the user.

    Args
        str_repr -- str, string representation set by the user
        value -- float, value of the unit in SI units
    """

    def __new__(self, str_repr, value):
        return float.__new__(self, value)

    def __init__(self, str_repr, value):
        float.__init__(value)
        self._str_repr = str_repr

    def __str__(self):
        return self._str_repr


GRAVITATIONAL_CONSTANT = 6.674e-11
SOLAR_MASS = Unit(r'$M_{\mathrm{\odot}}$', 1.98847e30)
JUPITER_MASS = Unit(r'$M_{\mathrm{J}}$', 1.898e27)
EARTH_MASS = Unit(r'$M_{\mathrm{oplus}}', 5.972e24)
AU = Unit('au', 1.496e8)
DAY = Unit('day', 86400.)
YEAR = Unit('year', 365. * DAY)
HOUR = Unit('hour', DAY / 24.)
KILOMETER = Unit('km', 1e3)
METER = Unit('m', 1.)
CENTIMETER = Unit('cm', 1e-2)
SECOND = Unit('s', 1.)
METER_PER_SECOND = Unit('m/s', 1.)

Position = namedtuple('Position', ['x', 'y'])
Observables = namedtuple('Observables', ['time', 'position', 'radial_velocity'])


def degree_to_radian(angle):
    """Convert angle in degree to radian."""
    return (angle * np.pi)/180.

def position_to_xy(position):
    """Converts a list of positions into two lists x and y."""
    return map(list, zip(*position))

def repr_quantity(quantity, unit):
    """Returns string of quantity + unit seperated by a space."""
    return ' '.join((str(quantity), '['+str(unit)+']'))


class OrbitAnimation:
    """Class used to create orbital animation."""

    def __init__(self, orbit, **SETTINGS):
        """Initialize the attributes used for the orbit animation.

        Args
            orbit -- the orbit object for which to make the animation
            SETTINGS -- contains the settings for the plot, defaults are below
        """

        self.X_UNIT = SETTINGS.pop('X_UNIT', AU)
        self.Y_UNIT = SETTINGS.pop('Y_UNIT', AU)
        self.TIME_UNIT = SETTINGS.pop('TIME_UNIT', DAY)
        self.RADIAL_VELOCITY_UNIT = SETTINGS.pop('RADIAL_VELOCITY_UNIT',
                                                  METER_PER_SECOND)
        self.DXLIM = SETTINGS.pop('DXLIM', 0.1)
        self.DYLIM = SETTINGS.pop('DYLIM', 0.1)
        self.DVRLIM = SETTINGS.pop('DVRLIM', 0.1)
        self.FONTSIZE = SETTINGS.pop('FONTSIZE', 13)

        self.fig = plt.figure()
        gs = self.fig.add_gridspec(3, 1, hspace=0.6, left = 0.15, right = 0.95)
        self.ax1 = self.fig.add_subplot(gs[0:2, 0])
        self.ax2 = self.fig.add_subplot(gs[2, 0])

        self.orbit = orbit
        self.body = self.ax1.scatter([], [], sizes = [50], color = [TABLEAU20[0]])
        self.line, = self.ax2.plot([], [], color = TABLEAU20[0])

    def ani_init(self):
        """Initalize animation."""
        time, x, y, radial_velocity = self.orbit.history
        y /= self.Y_UNIT
        x /= self.X_UNIT
        time /= self.TIME_UNIT
        radial_velocity /= self.RADIAL_VELOCITY_UNIT

        self.line.set_data([], [])
        self.body.set_offsets([])

        self.ax1.plot(x, y, color = 'k', lw = 1)
        self.ax1.set_xlim(x.min() - self.DXLIM, x.max() + self.DXLIM)
        self.ax1.set_ylim(y.min() - self.DXLIM, y.max() + self.DXLIM)
        self.ax1.set_xlabel(repr_quantity(r'X', self.X_UNIT),
                                          fontsize = self.FONTSIZE)
        self.ax1.set_ylabel(repr_quantity(r'Y', self.Y_UNIT),
                                          fontsize = self.FONTSIZE)
        self.ax2.set_xlabel(repr_quantity(r'Time', self.TIME_UNIT),
                                          fontsize = self.FONTSIZE)
        self.ax2.set_ylabel(repr_quantity(r'$v_{\mathrm{r}}$', self.RADIAL_VELOCITY_UNIT), fontsize = self.FONTSIZE)
        self.ax2.set_xlim(time.min(), time.max() + self.DVRLIM)
        self.ax2.set_ylim(radial_velocity.min() - self.DVRLIM, radial_velocity.max() + self.DVRLIM)

        return self.body, self.line,

    def ani_func(self, i):
        """Animation function."""
        time, x, y, radial_velocity = self.orbit.history
        x /= self.X_UNIT
        y /= self.Y_UNIT
        time /= self.TIME_UNIT
        radial_velocity /= self.RADIAL_VELOCITY_UNIT

        self.line.set_data(time[:i], radial_velocity[:i])
        self.body.set_offsets((x[i], y[i]))

        return self.body, self.line,

    def start(self):
        """Start orbital animation."""
        NFRAMES = len(self.orbit._history)
        ani = animation.FuncAnimation(self.fig, self.ani_func, frames = NFRAMES,
                                      interval = 10, blit = True,
                                      init_func = self.ani_init)

        plt.show()


class Orbit:
    """Class used to create Keplerian orbits.

    EXAMPLE
        Create orbit, evolve it and finally create an animation of
        the orbital history.

        from systems import earth

        orbit = Orbit('Earth', **earth)
        orbit.evolve(time = orbit.period)
        orbit.animate()
    """

    def __init__(self, body_name, **orbital_params):
        """Initialize the orbit."""
        self.body_name = body_name

        self.mass1 = orbital_params.pop('mass1', 1.)
        self.mass2 = orbital_params.pop('mass2', 1.)
        self.system_velocity = orbital_params.pop('system_velocity', 0.)
        self.semi_major_axis = orbital_params.pop('semi_major_axis', 1.)
        self.ecc = orbital_params.pop('ecc', 0.)
        self.incl = orbital_params.pop('incl', 90.)
        self.omega = orbital_params.pop('omega', 0.)
        self.Omega = orbital_params.pop('Omega', 0.)
        self.t0 = orbital_params.pop('t0', 0.)
        self.period = orbital_params.pop('period', 1.)
        self.time = orbital_params.pop('time', 0.)

        # Update the non-public parameters as
        # they depend on the public parameters
        self._update()
        self._history = []

    @property
    def observables(self):
        """Return a namedtuple of the observables."""
        return Observables(self.time, self.position, self.radial_velocity)

    @property
    def incl(self):
        """Return the inclination in radians"""
        return self._incl

    @incl.setter
    def incl(self, value):
        self._incl = degree_to_radian(value)

    @property
    def omega(self):
        """Return the argument of periapsis (omega) in radian."""
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = degree_to_radian(value)

    @property
    def Omega(self):
        """Return the longitude of periapsis (Omega) in radian."""
        return self._Omega

    @Omega.setter
    def Omega(self, value):
        self._Omega = degree_to_radian(value)

    @property
    def r(self):
        """Return the distance between body 1 and body 2."""
        return self._r
    
    def _update(self):
        """Update the non-public parameters."""
        self._mean_anomaly = ((2. * np.pi)/self.period) * (self.time - self.t0)
        self._ecc_anomaly = self._solve_kepler_equation()
        self._r = self.semi_major_axis * (1. - self.ecc * np.cos(self._ecc_anomaly))
        self._true_anomaly = 2. * np.arctan( np.sqrt((1.+self.ecc)/(1.-self.ecc)) * \
                                  np.tan(self._ecc_anomaly/2.))

    def _solve_kepler_equation(self, epsilon = 1e-3):
        """Solve the Kepler Equation for the eccentric anomly E.

        Args
            epsilon -- float, the precision with which the equation is solved

        Returns
            E -- float, the eccentric anomaly
        """
        E = float(self._mean_anomaly)
        solved = False
        while not solved:
            E_new_estimate = self._mean_anomaly + self.ecc * np.sin(E)
            E, residual = E_new_estimate, abs(E_new_estimate - E)
            solved = (residual < epsilon) # The Kepler Equation is considered
                                          # solved when the left-hand side
                                          # and the right-hand side differ less
                                          # than epsilon
        return E

    @property
    def position(self):
        """Return the position (x, y) on the sky.

        Returns
            Position -- a namedtuple (x, y)
        """
        x = self.r \
            * ((np.cos(self.Omega) * np.cos(self.omega + self._true_anomaly)) \
            - (np.sin(self.Omega) * np.sin(self.omega + self._true_anomaly) \
            * np.cos(self.incl)) )
        y = self.r \
            * ((np.sin(self.Omega) * np.cos(self.omega + self._true_anomaly)) \
            + (np.cos(self.Omega) * np.sin(self.omega + self._true_anomaly) \
            * np.cos(self.incl)) )
        return Position(x, y)

    @property
    def K(self):
        """Return the semi-amplitude of the radial velocity."""
        return ((2.*np.pi*GRAVITATIONAL_CONSTANT) / self.period)**(1./3.) \
            * ((self.mass1) / ((self.mass1+self.mass2)**(2./3.))) \
            * (1./(1. - self.ecc**2)**(1./2.)) \
            * np.sin(self.incl)


    @property
    def radial_velocity(self):
        """Return the radial velocity."""
        return self.system_velocity \
            + self.K * (np.cos(self.omega + self._true_anomaly) \
            + self.ecc * np.cos(self.omega))

    @property
    def history(self):
        """Returns the orbital history of the system."""
        time, position, radial_velocity = map(np.array, zip(*self._history))
        x, y = map(np.array, zip(*position))
        return time, x, y, radial_velocity

    def clear_history(self):
        """Clear the history of the system."""
        self._history = []

    def step(self, dt):
        """Step by a timestep dt."""
        self.time += dt
        self._update()

    def evolve(self, time, dt = None, mode='dt'):
        """Evolve orbit over time by timesteps dt."""
        if mode == 'dt':
            if dt is None:
                DT_DEFAULT = self.period/500. # ensures a smooth orbit when
                dt = DT_DEFAULT               # dt is not defined by the user

            time_left = time
            while time_left > 0.:
                self.step(dt)
                self._history.append(self.observables)
                time_left -= dt
        elif mode == 'time':
            self.time = time[0]
            #self._history.append(self.observables)
            dt_list = np.insert(1e-99, 0, np.diff(time))
            for dt in dt_list:
                self.step(dt)
                self._history.append(self.observables)

    def plot(self):
        """Plot the observables in the orbital history."""
        fig = plt.figure()
        gs = fig.add_gridspec(3, 1)
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax2 = fig.add_subplot(gs[2, 0])

        time, x, y, radial_velocity = orbit.history
        ax1.plot(x / AU, y / AU)
        #         ax1.set_xlim(-1, 1)
        #         ax1.set_ylim(-1, 1)
        ax2.plot(time, radial_velocity)

        plt.show()

    def animate(self, **SETTINGS):
        """Animate the orbital history."""
        animation = OrbitAnimation(self, **SETTINGS)
        animation.start()