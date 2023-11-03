import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io.fits import getdata
from scipy.interpolate import interp2d

import exoverses.base as base


class ExovistaStar(base.star.Star):
    """
    Class for the star in the exoVista systems
    """

    def __init__(self, infile):
        # Get the object's data from the fits file
        with open(infile, "rb") as f:
            obj_data, obj_header = getdata(f, ext=4, header=True, memmap=False)
            self.ev_wavelengths = getdata(f, ext=0, header=False, memmap=False) * u.um

        # The times that the exovista scene was generated at
        self.ev_t = obj_data[:, 0] * u.yr

        # Positions at times the exovista scene was generated at
        self.ev_x = obj_data[:, 9] * u.AU
        self.ev_y = obj_data[:, 10] * u.AU
        self.ev_z = obj_data[:, 11] * u.AU

        # Velocities at times the exovista scene was generated at
        self.ev_vx = obj_data[:, 12] * u.AU / u.yr
        self.ev_vy = obj_data[:, 13] * u.AU / u.yr
        self.ev_vz = obj_data[:, 14] * u.AU / u.yr

        # Load star's spectral flux density in Janskys
        self.ev_star_flux_density = np.array(obj_data[:, 16:])
        self.star_flux_density_interp = interp2d(
            self.ev_wavelengths, self.ev_t, self.ev_star_flux_density, kind="quintic"
        )

        # System identifiers
        self.id = obj_header["ID"]
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
        self.MV = obj_header["M_V"] * u.mag  # Absolute V band mag

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
        # Bolometric luminosity
        self.luminosity = obj_header["LSTAR"] * u.Lsun
        self.effective_temperature = obj_header["TEFF"] * u.K
        self.angular_diameter = obj_header["ANGDIAM"] * u.mas
        self.mass = obj_header["MASS"] * u.M_sun
        self.radius = obj_header["RSTAR"] * u.R_sun
        self.logg = obj_header["LOGG"] * u.cm / u.s**2
        self.mu = self.mass * const.G
        self.pixel_scale = obj_header["PXSCLMAS"] * u.mas / u.pixel

        # Propagation table
        self.vectors = pd.DataFrame(
            {
                "t": [self.ev_t[0].decompose().value],
                "x": [self.ev_x[0].decompose().value],
                "y": [self.ev_y[0].decompose().value],
                "z": [self.ev_z[0].decompose().value],
                "vx": [self.ev_vx[0].decompose().value],
                "vy": [self.ev_vy[0].decompose().value],
                "vz": [self.ev_vz[0].decompose().value],
            }
        )

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
        return self.star_flux_density_interp(wavelengths, times) * u.Jy
