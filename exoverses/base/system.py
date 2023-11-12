import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time
from keplertools import fun as kt


class System:
    """
    Class for a single system. Must have a star and a list of planets.
    """

    def __init__(self, star=None, planets=None, disk=None) -> None:
        self.star = star
        self.planets = planets
        self.disk = disk
        if self.planets is not None:
            self.pInds = np.arange(len(self.planets))
            self.cleanup()

    def __repr__(self):
        return (
            f"{self.star.name}\tdist:{self.star.dist}\t"
            f"Type:{self.star.spectral_type}\n\n"
            f"Planets:\n{self.get_p_df()}"
        )

    def cleanup(self):
        # Sort the planets in the system by semi-major axis
        a_vals = [planet.a.value for planet in self.planets]
        self.planets = np.array(self.planets)[np.argsort(a_vals)].tolist()
        self.pInds = self.pInds[np.argsort(a_vals)]

    def getpattr(self, attr):
        # Return array of all planet's attribute value, e.g. all semi-major
        # axis values
        if type(getattr(self.planets[0], attr)) == u.Quantity:
            return [getattr(planet, attr).value for planet in self.planets] * getattr(
                self.planets[0], attr
            ).unit
        else:
            return [getattr(planet, attr) for planet in self.planets]

    def get_p_df(self):
        patts = [
            "K",
            "T",
            "secosw",
            "sesinw",
            "T_c",
            "a",
            "e",
            "inc",
            "W",
            "w",
            "M0",
            "t0",
            "mass",
            "radius",
        ]
        p_df = pd.DataFrame()
        for att in patts:
            pattr = self.getpattr(att)
            if type(pattr) == u.Quantity:
                p_df[att] = pattr.value
            else:
                p_df[att] = pattr

        return p_df

    def propagate(self, times):
        """
        Propagates system at all times given. Currently does not handle
        """
        # Get unique time values, multiple instruments can be scheduled to
        # observe at the same time
        times = Time(np.unique(times.jd), format="jd")

        syst_M = []
        for planet in self.planets:
            syst_M.append(planet.mean_anom(times))
        Marr = np.stack(syst_M).value
        rv_vals = (
            -kt.calc_RV_from_M(
                Marr,
                self.getpattr("e"),
                self.getpattr("w").to(u.rad).value,
                self.getpattr("K").value,
            )
            * u.m
            / u.s
        ).T
        df = pd.DataFrame(rv_vals)
        df["rv"] = df.sum(axis=1)
        df["t"] = times

        # Storing as dataframe too

        # rv_df = pd.DataFrame(
        #     np.stack((times, rv_vals.value), axis=-1), columns=["t", "rv"]
        # )
        # self.rv_df = rv_df
        return df
