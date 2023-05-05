import astropy.units as u
import pandas as pd

from exoverses.base.planet import Planet


class ExosimsPlanet(Planet):
    """
    Class for the planets in the EXOSIMS systems
    Assumes e=0, which is the case for all downloaded systems
    """

    def __init__(self, SU, star, pInd, t0):
        self.star = star

        # Time data
        self._t = t0.decimalyear * u.yr
        self.t0 = t0

        # Position data
        self._x = SU.r[pInd][0]
        self._y = SU.r[pInd][1]
        self._z = SU.r[pInd][2]

        # Velocity data
        self._vx = SU.v[pInd][0]
        self._vy = SU.v[pInd][1]
        self._vz = SU.v[pInd][2]

        # Assign the planet's keplerian orbital elements
        # self.a = SU.a[pInd]
        # self.e = SU.e[pInd]
        # self.inc = SU.I[pInd]
        # self.W = SU.O[pInd]
        # self.w = SU.w[pInd]

        # # Assign the planet's mass/radius information
        # self.mass = SU.Mp[pInd]
        # self.radius = SU.Rp[pInd]
        # # Initial mean anomaly
        # self.M0 = SU.M0[pInd]

        planet_dict = {
            "t0": t0,
            "a": SU.a[pInd],
            "e": SU.e[pInd],
            "inc": SU.I[pInd],
            "W": SU.O[pInd],
            "w": SU.w[pInd],
            "mass": SU.Mp[pInd],
            "radius": SU.Rp[pInd],
            "M0": SU.M0[pInd],
            "p": 0.2,
        }
        Planet.__init__(self, planet_dict)
        self.solve_dependent_params()
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
