import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io.fits import getdata
from astropy.time import Time
from scipy.interpolate import interp1d, interp2d

import exoverses.base as base


class ExovistaPlanet(base.planet.Planet):
    """
    Class for the planets in the exoVista systems
    Assumes e=0, which is the case for all downloaded systems
    """

    def __init__(self, infile, fits_ext, star):
        # Get the object's data from the fits file
        with open(infile, "rb") as f:
            obj_data, obj_header = getdata(f, ext=fits_ext, header=True, memmap=False)
            self._wavelengths = getdata(f, ext=0, header=False, memmap=False) * u.um

        self.star = star
        # Time data, setting default epoch to the year 2000
        self._t = Time(2000 + obj_data[:, 0], format="decimalyear")
        self.t0 = self._t[0]

        self._x_pix = obj_data[:, 1] * u.pixel
        self._y_pix = obj_data[:, 2] * u.pixel
        _x_pix_interp = interp1d(self._t.jd, self._x_pix.value, kind="cubic")
        _y_pix_interp = interp1d(self._t.jd, self._y_pix.value, kind="cubic")
        self._x_pix_interp = lambda t: _x_pix_interp(t.jd)
        self._y_pix_interp = lambda t: _y_pix_interp(t.jd)

        # Barycentric position data
        self._x = obj_data[:, 9] * u.AU
        self._y = obj_data[:, 10] * u.AU
        self._z = obj_data[:, 11] * u.AU

        # Barycentric velocity data
        self._vx = obj_data[:, 12] * u.AU / u.yr
        self._vy = obj_data[:, 13] * u.AU / u.yr
        self._vz = obj_data[:, 14] * u.AU / u.yr

        # Assign the planet's time-varying mean anomaly, argument of pericenter,
        # true anomaly, and contrast
        self.rep_w = (obj_data[:, 7] * u.deg + 180 * u.deg) % (2 * np.pi * u.rad)
        self.M = (obj_data[:, 8] * u.deg + 180 * u.deg) % (2 * np.pi * u.rad)
        self.nu = (self.rep_w + self.M) % (
            2 * np.pi * u.rad
        )  # true anomaly for circular orbits
        self.phase_angles = obj_data[:, 15]

        self.contrast = obj_data[:, 16:]
        self.contrast_interp = interp2d(
            self._wavelengths,
            self._t.decimalyear * u.yr,
            self.contrast,
            kind="quintic",
        )

        # Spectral flux density of the planet
        self.planet_spec_flux_density_interp = interp2d(
            self._wavelengths,
            self._t.decimalyear * u.yr,
            np.multiply(self.contrast, star._star_flux_density),
            kind="quintic",
        )

        # Initial mean anomaly
        self.M0 = self.nu[0]
        planet_dict = {
            "t0": self.t0,
            "a": obj_header["A"] * u.AU,
            "e": obj_header["E"],
            "inc": (obj_header["I"] * u.deg).to(u.rad) + star.midplane_I,
            "W": (obj_header["LONGNODE"] * u.deg).to(u.rad),
            "w": 0 * u.rad,
            "mass": obj_header["M"] * u.M_earth,
            "radius": obj_header["R"] * u.R_earth,
            "M0": self.M0,
            "p": 0.2,
        }
        base.planet.Planet.__init__(self, planet_dict)
        self.solve_dependent_params()

        # Assign the planet's keplerian orbital elements
        # self.a = obj_header["A"] * u.AU
        # self.e = obj_header["E"]
        # self.inc = (obj_header["I"] * u.deg).to(u.rad)
        # self.W = (obj_header["LONGNODE"] * u.deg).to(u.rad)
        # # self.w = (obj_header["ARGPERI"] * u.deg).to(u.rad)
        # self.w = 0 * u.rad

        # # Assign the planet's mass/radius information
        # self.mass = obj_header["M"] * u.M_earth
        # self.radius = obj_header["R"] * u.R_earth

        # # Gravitational parameter
        # self.mu = (const.G * (self.mass + star.mass)).decompose()
        # self.T = (2 * np.pi * np.sqrt(self.a**3 / self.mu)).to(u.d)
        # self.w_p = self.w
        # self.w_s = (self.w + np.pi * u.rad) % (2 * np.pi * u.rad)
        # self.secosw = np.sqrt(self.e) * np.cos(self.w)
        # self.sesinw = np.sqrt(self.e) * np.sin(self.w)

        # Because we have the mean anomaly at an epoch we can calculate the
        # time of periastron as t0 - T_e where T_e is the time since periastron
        # passage
        # T_e = (self.T * self.M0 / (2 * np.pi * u.rad)).decompose()
        # self.T_p = self.t0 - T_e

        # Calculate the time of conjunction
        # self.T_c = Time(
        #     rvo.timeperi_to_timetrans(
        #         self.T_p.jd, self.T.value, self.e, self.w_s.value
        #     ),
        #     format="jd",
        # )
        # self.K = (
        #     (2 * np.pi * const.G / self.T) ** (1 / 3.0)
        #     * (self.mass * np.sin(self.inc) / star.mass ** (2 / 3.0))
        #     * (1 - self.e**2) ** (-1 / 2)
        # ).decompose()

        # # Mean angular motion
        # self.n = (np.sqrt(self.mu / self.a**3)).decompose() * u.rad

        # Propagation table
        self.vectors = pd.DataFrame(
            {
                "t": [self._t[0].decimalyear],
                "x": [self._x[0].decompose().value],
                "y": [self._y[0].decompose().value],
                "z": [self._z[0].decompose().value],
                "vx": [self._vx[0].decompose().value],
                "vy": [self._vy[0].decompose().value],
                "vz": [self._vz[0].decompose().value],
            }
        )

        self.star = star

        # self.classify_planet()

    def spec_flux_density(self, wavelengths, times):
        """
        Calculate the spectral flux density of the star at the given wavelengths
        and times
        Args:
            wavelengths (astropy Quantity array):
                Wavelengths to calculate spectral flux density
            times (astropy Time array):
                Times to calculate spectral flux density
        Returns:
            F (astropy Quantity array):
                Spectral flux density values
        """
        return self.planet_spec_flux_density_interp(wavelengths, times) * u.Jy
