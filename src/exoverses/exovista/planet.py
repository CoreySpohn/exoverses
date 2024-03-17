import astropy.units as u
import numpy as np
from astropy.io.fits import getdata
from astropy.time import Time
from scipy.interpolate import interp2d

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

        # self._x_pix = obj_data[:, 1] * u.pixel
        # self._y_pix = obj_data[:, 2] * u.pixel
        # _x_pix_interp = interp1d(self._t.jd, self._x_pix.value, kind="cubic")
        # _y_pix_interp = interp1d(self._t.jd, self._y_pix.value, kind="cubic")
        # self._x_pix_interp = lambda t: _x_pix_interp(t.jd)
        # self._y_pix_interp = lambda t: _y_pix_interp(t.jd)

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
        self.rep_w = (obj_data[:, 7] * u.deg) % (2 * np.pi * u.rad)
        self.M = (obj_data[:, 8] * u.deg) % (2 * np.pi * u.rad)
        # true anomaly for circular orbits
        self.nu = (self.rep_w + self.M) % (2 * np.pi * u.rad)
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
        self.M0 = self.M[0]
        planet_dict = {
            "t0": self.t0,
            "a": obj_header["A"] * u.AU,
            "e": obj_header["E"],
            "inc": (obj_header["I"] * u.deg),
            "W": (obj_header["LONGNODE"] * u.deg),
            "w": (obj_header["ARGPERI"] * u.deg),
            "mass": obj_header["M"] * u.M_earth,
            "radius": obj_header["R"] * u.R_earth,
            "M0": self.M0,
            "p": 0.2,
        }
        base.planet.Planet.__init__(self, planet_dict, star)
        self.solve_dependent_params()

        self.star = star

        self.classify_planet()

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
        return (
            self.planet_spec_flux_density_interp(wavelengths, times.decimalyear) * u.Jy
        )

    # def rotate_to_sky_coords(self, vec, roll_angle=0 * u.rad):
    #     """
    #     Rotate from barycentric coordinates to plane of the sky, this is set up
    #     to match the exovista data

    #     Args:
    #         vec (np.array):
    #             Nx3 array of vectors in system-plane coordinates
    #         roll_angle (astropy Quantity):
    #             Angle to rotate the vectors by in the plane of the sky to simulate
    #             roll of the telescope

    #     Returns:
    #         vec (np.array):
    #             Nx3 array of vectors rotated to sky coordinates

    #     """
    #     # Rotate around x axis with midplane inclination
    #     vec = misc.rotate_vectors(vec.T, [1, 0, 0], -self.star.midplane_I)

    #     # Rotate around z axis with midplane position angle
    #     vec = misc.rotate_vectors(vec, [0, 0, 1], self.star.midplane_PA)

    #     # Flip around z axis
    #     vec[:, 2] = -vec[:, 2]

    #     if roll_angle != 0 * u.rad:
    #         # Rotate around y axis with roll angle
    #         vec = misc.rotate_vectors(vec, [0, 0, 1], roll_angle)
    #     return vec
