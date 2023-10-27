import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.time import Time

import exoverses.util.misc as misc
from exoverses.base.planet import Planet


class FitPlanet(Planet):
    def __init__(self, planet_dict, true_system):
        """
        filling in parameters that are not found in the fitting process.
        Assumes that the planet_dict includes
        T, Tc, secosw, sesinw, K
        basis.
        """
        self.star = true_system.star
        for att, value in planet_dict.items():
            setattr(self, att, value)
        self.e = self.secosw**2 + self.sesinw**2
        self.w_s = np.arctan2(self.sesinw, self.secosw) * u.rad
        self.w = (self.w_s + np.pi * u.rad) % (2 * np.pi * u.rad)
        self.w_p = self.w
        self.T_p = misc.timetrans_to_timeperi(
            self.T_c.jd, self.T.to(u.d).value, self.e, self.w_s.to(u.rad).value
        )
        self.msini = (
            misc.Msini(
                self.K.decompose().value,
                self.T.to(u.d).value,
                self.star.mass.to(u.M_sun).value,
                self.e,
            )
            * u.M_earth
        )
        self.mu = (const.G * self.star.mass).decompose()
        self.a = ((self.mu * (self.T / (2 * np.pi)) ** 2) ** (1 / 3)).decompose()

        # Finding the mean anomaly at time of conjunction
        nu_p = (np.pi / 2 * u.rad - self.w_s) % (2 * np.pi * u.rad)
        E_p = 2 * np.arctan2(
            np.sqrt((1 - self.e)) * np.tan(nu_p / 2), np.sqrt((1 + self.e))
        )
        self.M0 = (E_p - self.e * np.sin(E_p) * u.rad) % (2 * np.pi * u.rad)
        self.t0 = Time(self.T_c.jd, format="jd")

        self.compare_to_system(true_system)

    def compare_to_system(self, true_system):
        """
        Function to check how well a planet fit compares to the true system.
        Gets the index of the planet that the fit is most likely describing
        and the RMS error of the fit.
        """
        # Not doing argument of periastron because it is finicky for circular
        # orbits
        # werr = true_system.getpattr('w') - fitted_planet.w
        eerr = true_system.getpattr("e") - self.e
        Kerr = true_system.getpattr("K") - self.K
        Terr = true_system.getpattr("T") - self.T

        # Normalize the differences by the range of the system values
        Knorm = Kerr / (np.ptp(true_system.getpattr("K")))
        Tnorm = Terr / (np.ptp(true_system.getpattr("T")))

        self.all_rms = np.sqrt(eerr**2 + Knorm**2 + Tnorm**2)
        self.best_match = np.argmin(self.all_rms)
        self.best_rms = self.all_rms[self.best_match]
