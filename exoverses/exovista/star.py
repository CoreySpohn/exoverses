import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io.fits import getdata

import exoverses.base as base


class ExovistaStar(base.star.Star):
    """
    Class for the star in the exoVista systems
    """

    def __init__(self, infile):
        # Get the object's data from the fits file
        with open(infile, "rb") as f:
            obj_data, obj_header = getdata(f, ext=3, header=True, memmap=False)

        # Time data
        self._t = obj_data[:, 0] * u.yr

        # Position data
        self._x = obj_data[:, 9] * u.AU
        self._y = obj_data[:, 10] * u.AU
        self._z = obj_data[:, 11] * u.AU

        # Velocity data
        self._vx = obj_data[:, 12] * u.AU / u.yr
        self._vy = obj_data[:, 13] * u.AU / u.yr
        self._vz = obj_data[:, 14] * u.AU / u.yr

        # System identifiers
        self.id = obj_header["STARID"]
        self.name = f"HIP {obj_header['HIP']}"

        # System midplane information
        self.midplane_PA = (obj_header["PA"] * u.deg).to(u.rad)  # Position angle
        self.midplane_I = np.abs((obj_header["I"] * u.deg).to(u.rad))  # Inclination
        # if self.midplane_I < 0:
        #     breakpoint()

        # Proper motion
        self.PMRA = obj_header["PMRA"] * u.mas / u.yr
        self.PMDEC = obj_header["PMDEC"] * u.mas / u.yr

        # Celestial coordinates
        self.RA = obj_header["RA"] * u.deg
        self.DEC = obj_header["DEC"] * u.deg
        self.dist = obj_header["DIST"] * u.pc
        self.coords = SkyCoord(ra=self.RA, dec=self.DEC, distance=self.dist)

        # Spectral properties
        self.spectral_type = obj_header["TYPE"]
        self.MV = obj_header["M_V"]  # Absolute V band mag

        # Commenting out for now, these are available but
        # not every star has all the information
        # self.Bmag = obj_header["BMAG"]
        # self.Vmag = obj_header["VMAG"]
        # self.Rmag = obj_header["RMAG"]
        # self.Imag = obj_header["IMAG"]
        # self.Jmag = obj_header["JMAG"]
        # self.Hmag = obj_header["HMAG"]
        # self.Kmag = obj_header["KMAG"]

        # Stellar properties
        self.Lstar = obj_header["LSTAR"] * u.Lsun  # Bolometric luminosity
        self.Teff = obj_header["TEFF"] * u.K  # Effective temperature
        self.angdiam = obj_header["ANGDIAM"]  # Angular diameter
        self.mass = obj_header["MASS"] * u.M_sun
        self.radius = obj_header["RSTAR"] * u.R_sun
        self.mu = self.mass * const.G

        # Propagation table
        self.vectors = pd.DataFrame(
            {
                "t": [self._t[0].decompose().value],
                "x": [self._x[0].decompose().value],
                "y": [self._y[0].decompose().value],
                "z": [self._z[0].decompose().value],
                "vx": [self._vx[0].decompose().value],
                "vy": [self._vy[0].decompose().value],
                "vz": [self._vz[0].decompose().value],
            }
        )
