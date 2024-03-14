import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time
from keplertools import fun as kt

import exoverses.util.misc as misc


class Planet:
    """
    Class for a planet
    """

    def __init__(self, planet_dict, star) -> None:
        for att, value in planet_dict.items():
            setattr(self, att, value)
        self.star = star
        self.solve_dependent_params()

    def __repr__(self):
        """
        Make dataframe with planet attributes
        """
        params = self.dump_params()
        res = {}
        for key, val in params.items():
            if type(val) == u.Quantity:
                res[key] = val.value
            elif type(val) == Time:
                res[key] = val.decimalyear
            else:
                res[key] = val

        # Create dataframe from res dictionary
        p_df = pd.DataFrame(res, index=[0])

        return f"{type(self).__name__} object\n{p_df}"

    def dump_params(self):
        params = {
            "t0": self.t0,
            "a": self.a,
            "e": self.e,
            "mass": self.mass,
            "radius": self.radius,
            "inc": self.inc,
            "W": self.W,
            "w": self.w,
            "M0": self.M0,
            "p": self.p,
        }
        return params

    def calc_vectors(
        self,
        t,
        return_r=True,
        return_v=False,
        coord_system="barycentric",
        convention="exovista",
    ):
        """
        Given a time, calculate the planet's barycentric position and/or
        velocity vectors
        Args:
            t (Time):
                Time to calculate the position vectors at
            return_r (bool):
                Whether to return the position vectors
            return_v (bool):
                Whether to return the velocity vectors
            coord_system (str):
                Coordinate system to return vectors in, either "barycentric" or
                "sky"
            convention (str):
                Orbital convention for rotations
        Returns:
            r(astropy Quantity array):
                3 x n stacked position vector in meters
            v(astroypy Quantity array):
                3 x n stacked velocity vector in m/s

        """
        # This will find the radial and velocity vectors at an epoch
        M = self.mean_anom(t)
        E = kt.eccanom(M.to(u.rad).value, self.e)
        orb_elem = (
            self.a.decompose().value,
            self.e,
            self.W.to(u.rad).value,
            self.inc.to(u.rad).value,
            self.w.to(u.rad).value,
        )
        a, e, Omega, inc, w = orb_elem
        if not np.isscalar(E):
            a = np.ones(len(E)) * a
            e = np.ones(len(E)) * e
            Omega = np.ones(len(E)) * Omega
            inc = np.ones(len(E)) * inc
            w = np.ones(len(E)) * w

        sinw = np.sin(w)
        cosw = np.cos(w)
        sinO = np.sin(Omega)
        cosO = np.cos(Omega)
        sininc = np.sin(inc)
        cosinc = np.cos(inc)
        asqrt1me2 = a * np.sqrt(1 - e**2)

        A = np.vstack(
            (
                a * (cosO * cosw - sinO * cosinc * sinw),
                a * (sinO * cosw + cosO * cosinc * sinw),
                a * sininc * sinw,
            )
        )

        B = np.vstack(
            (
                -asqrt1me2 * (cosO * sinw + sinO * cosinc * cosw),
                asqrt1me2 * (-sinO * sinw + cosO * cosinc * cosw),
                asqrt1me2 * sininc * cosw,
            )
        )

        # Calculate position vectors
        if return_r:
            if np.isscalar(self.mu) and not (np.isscalar(E)):
                r = np.matmul(A, np.array((np.cos(E) - e), ndmin=2)) + np.matmul(
                    B, np.array(np.sin(E), ndmin=2)
                )
            else:
                r = np.matmul(A, np.diag(np.cos(E) - e)) + np.matmul(
                    B, np.diag(np.sin(E))
                )
            if coord_system == "sky":
                r = misc.rotate_to_sky_coords(
                    r, self.star.midplane_I, self.star.midplane_PA, convention
                ).T
            r *= u.m

        # Calculate velocity vectors
        if return_v:
            if np.isscalar(self.mu) and not (np.isscalar(E)):
                v = (
                    np.matmul(-A, np.array(np.sin(E), ndmin=2))
                    + np.matmul(B, np.array(np.cos(E), ndmin=2))
                ) * np.tile(
                    np.sqrt(self.mu.decompose().value * a ** (-3.0))
                    / (1 - e * np.cos(E)),
                    (3, 1),
                )
            else:
                v = np.matmul(
                    np.matmul(-A, np.diag(np.sin(E)))
                    + np.matmul(B, np.diag(np.cos(E))),
                    np.diag(
                        np.sqrt(self.mu.decompose().value * a ** (-3.0))
                        / (1 - e * np.cos(E))
                    ),
                )
            if coord_system == "sky":
                v = misc.rotate_to_sky_coords(
                    v, self.star.midplane_I, self.star.midplane_PA, convention
                ).T

            v *= u.m / u.s

        if return_r and return_v:
            return r, v
        if return_r:
            return r
        if return_v:
            return v

    def rotate_to_sky_coords(self, vec):
        """
        Rotate the given vector to the sky coordinates, stub in the base class
        """
        return vec

    def mean_anom(self, times):
        """
        Calculate the mean anomaly at the given times
        Args:
            times (astropy Time array):
                Times to calculate mean anomaly

        Returns:
            M (astropy Quantity array):
                Planet's mean anomaly at t (radians)
        """
        M = ((self.n * ((times.jd - self.t0.jd) * u.d)).decompose() + self.M0) % (
            2 * np.pi * u.rad
        )
        return M

    def solve_dependent_params(self):
        self.mu = (const.G * (self.mass + self.star.mass)).decompose()
        self.T = (2 * np.pi * np.sqrt(self.a**3 / self.mu)).to(u.d)
        self.w_p = self.w
        self.w_s = (self.w + np.pi * u.rad) % (2 * np.pi * u.rad)
        self.secosw = np.sqrt(self.e) * np.cos(self.w)
        self.sesinw = np.sqrt(self.e) * np.sin(self.w)
        T_e = (self.T * self.M0 / (2 * np.pi * u.rad)).decompose()
        self.T_p = self.t0 - T_e

        # Calculate the time of conjunction
        self.T_c = Time(
            misc.timeperi_to_timetrans(
                self.T_p.jd, self.T.value, self.e, self.w_s.value
            ),
            format="jd",
        )
        self.K = (
            (2 * np.pi * const.G / self.T) ** (1 / 3.0)
            * (self.mass * np.sin(self.inc) / self.star.mass ** (2 / 3.0))
            * (1 - self.e**2) ** (-1 / 2)
        ).decompose()

        # Mean angular motion
        self.n = (np.sqrt(self.mu / self.a**3)).decompose() * u.rad

    def classify_planet(self):
        """
        This determines the Kopparapu bin of the planet This is adapted from
        the EXOSIMS SubtypeCompleteness method classifyPlanets so that EXOSIMS
        isn't a mandatory import
        """
        # Reverse luminosity scaling
        a = self.a.to("AU").value / np.sqrt(self.star.luminosity.to("Lsun").value)

        lower_a = 0.95
        upper_a = 1.67

        lower_R = 0.8 / np.sqrt(a)
        upper_R = 1.4
        self.is_earth = (lower_a <= a < upper_a) and (
            lower_R <= self.radius.to("earthRad").value < upper_R
        )

        # # Calculate the luminosity of the star, assuming main-sequence
        # if self.mass < 2 * u.M_sun:
        #     self.Ls = const.L_sun * (self.star.mass / const.M_sun) ** 4
        # else:
        #     self.Ls = 1.4 * const.L_sun * (self.star.mass / const.M_sun) ** 3.5
        #
        # Rp = self.radius.to("earthRad").value
        # # a = self.a.to("AU").value
        # # e = self.e
        #
        # # Find the stellar flux at the planet's location as a fraction of earth's
        # earth_Lp = const.L_sun / (1 * (1 + (0.0167**2) / 2)) ** 2
        # self.Lp = (
        #     self.Ls / (self.a.to("AU").value * (1 + (self.e**2) / 2)) ** 2 / earth_Lp
        # )
        #
        # # Find Planet Rp range
        # Rp_bins = np.array([0, 0.5, 1.0, 1.75, 3.5, 6.0, 14.3, 11.2 * 4.6])
        # # Rp_lo = Rp_bins[:-1]
        # # Rp_hi = Rp_bins[1:]
        # Rp_types = [
        #     "Sub-Rocky",
        #     "Rocky",
        #     "Super-Earth",
        #     "Sub-Neptune",
        #     "Sub-Jovian",
        #     "Jovian",
        #     "Super-Jovian",
        # ]
        # self.L_bins = np.array(
        #     [
        #         [1000, 182, 1.0, 0.28, 0.0035, 5e-5],
        #         [1000, 182, 1.0, 0.28, 0.0035, 5e-5],
        #         [1000, 187, 1.12, 0.30, 0.0030, 5e-5],
        #         [1000, 188, 1.15, 0.32, 0.0030, 5e-5],
        #         [1000, 220, 1.65, 0.45, 0.0030, 5e-5],
        #         [1000, 220, 1.65, 0.40, 0.0025, 5e-5],
        #         [1000, 220, 1.68, 0.45, 0.0025, 5e-5],
        #         [1000, 220, 1.68, 0.45, 0.0025, 5e-5],
        #     ]
        # )
        # # self.L_bins = np.array(
        # #     [
        # #         [1000, 182, 1.0, 0.28, 0.0035, 5e-5],
        # #         [1000, 187, 1.12, 0.30, 0.0030, 5e-5],
        # #         [1000, 188, 1.15, 0.32, 0.0030, 5e-5],
        # #         [1000, 220, 1.65, 0.45, 0.0030, 5e-5],
        # #         [1000, 220, 1.65, 0.40, 0.0025, 5e-5],
        # #     ]
        # # )
        #
        # # Find the bin of the radius
        # self.Rp_bin = np.digitize(Rp, Rp_bins) - 1
        # try:
        #     self.Rp_type = Rp_types[self.Rp_bin]
        # except IndexError:
        #     print(f"Error handling Rp_type of planet with Rp_bin of {self.Rp_bin}")
        #     self.Rp_type = None
        #
        # # TODO Fix this to give correct when at edge cases since technically
        # # they're not straight lines
        #
        # # index of planet temp. cold,warm,hot
        # L_types = ["Very Hot", "Hot", "Warm", "Cold", "Very Cold"]
        # specific_L_bins = self.L_bins[self.Rp_bin, :]
        # self.L_bin = np.digitize(self.Lp.decompose().value, specific_L_bins) - 1
        # try:
        #     self.L_type = L_types[self.L_bin]
        # except IndexError:
        #     print(f"Error handling L_type of planet with L_bin of {self.L_bin}")
