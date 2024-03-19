import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io.fits import getdata
from astropy.time import Time
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
            self._wavelengths = getdata(f, ext=0, header=False, memmap=False) * u.um

        # The times that the exovista scene was generated at, assuming
        # it starts at J2000
        self._t = Time(2000 + obj_data[:, 0], format="decimalyear")

        # Interpolate the x position of the star
        # self._x_pix = obj_data[:, 1] * u.pixel
        # self._y_pix = obj_data[:, 2] * u.pixel
        # _x_pix_interp = interp1d(self._t.jd, self._x_pix.value, kind="cubic")
        # _y_pix_interp = interp1d(self._t.jd, self._y_pix.value, kind="cubic")
        # self._x_pix_interp = lambda t: _x_pix_interp(t.jd)
        # self._y_pix_interp = lambda t: _y_pix_interp(t.jd)

        # Positions at times the exovista scene was generated at
        self._x = obj_data[:, 9] * u.AU
        self._y = obj_data[:, 10] * u.AU
        self._z = obj_data[:, 11] * u.AU

        # Velocities at times the exovista scene was generated at
        self._vx = obj_data[:, 12] * u.AU / u.yr
        self._vy = obj_data[:, 13] * u.AU / u.yr
        self._vz = obj_data[:, 14] * u.AU / u.yr

        # Load star's spectral flux density in Janskys
        self._star_flux_density = np.array(obj_data[:, 16:])
        self.star_flux_density_interp = interp2d(
            self._wavelengths,
            self._t.decimalyear * u.yr,
            self._star_flux_density,
            kind="quintic",
        )

        # System identifiers
        self.id = obj_header["ID"]
        self.name = f"HIP {obj_header['HIP']}"

        # System midplane information
        # I don't know why the PA is negative, but to match the exovista
        # output positions it has to be
        self.midplane_PA = obj_header["PA"] * u.deg  # Position angle
        self.midplane_I = obj_header["I"] * u.deg  # Inclination
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

        self.calc_jitter_terms()

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
        return self.star_flux_density_interp(wavelengths, times.decimalyear) * u.Jy
