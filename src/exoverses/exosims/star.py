import astropy.constants as const
import astropy.units as u
import pandas as pd

from exoverses.base.star import Star


class ExosimsStar(Star):
    """
    Class for the star in the EXOSIMS systems
    """

    def __init__(self, SU, sInd):
        # Get the object's data from the fits file

        TL = SU.TargetList
        # Time data
        self._t = 0 * u.yr

        # Position data
        self._x = 0 * u.AU
        self._y = 0 * u.AU
        self._z = 0 * u.AU

        # Velocity data
        self._vx = 0 * u.AU / u.yr
        self._vy = 0 * u.AU / u.yr
        self._vz = 0 * u.AU / u.yr

        # System identifiers
        self.id = sInd
        self.name = TL.Name[sInd]

        # System midplane information
        # self.midplane_PA = (obj_header["PA"] * u.deg).to(u.rad)  # Position angle
        # self.midplane_I = (obj_header["I"] * u.deg).to(u.rad)  # Inclination

        # Proper motion
        self.PMRA = TL.pmra[sInd] * u.mas / u.yr
        self.PMDEC = TL.pmdec[sInd] * u.mas / u.yr

        # Celestial coordinates
        self.coords = TL.coords[sInd]
        self.RA = self.coords.ra
        self.DEC = self.coords.dec
        self.dist = TL.dist[sInd]

        # Spectral properties
        self.spectral_type = TL.spectral_class[sInd]
        self.MV = TL.Vmag[sInd]  # Absolute V band mag

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
        self.Lstar = TL.L[sInd] * u.Lsun  # Bolometric luminosity
        # self.Teff = obj_header["TEFF"] * u.K  # Effective temperature
        # self.angdiam = obj_header["ANGDIAM"] * u.K  # Angular diameter
        self.mass = TL.MsTrue[sInd]
        self.mu = self.mass * const.G
        # self.radius = obj_header["RSTAR"] * u.R_sun

        # Propagation table
        self.vectors = pd.DataFrame(
            {
                "t": [self._t.decompose().value],
                "x": [self._x.decompose().value],
                "y": [self._y.decompose().value],
                "z": [self._z.decompose().value],
                "vx": [self._vx.decompose().value],
                "vy": [self._vy.decompose().value],
                "vz": [self._vz.decompose().value],
            }
        )
