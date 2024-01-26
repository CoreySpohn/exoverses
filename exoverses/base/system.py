import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
import rebound
import xarray as xr
from astropy.time import Time
from keplertools import fun as kt
from tqdm import tqdm


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

    def propagate_img(self, times):
        """
        Propagates system at all times given. Currently does not handle
        """
        # Get unique time values, multiple instruments can be scheduled to
        # observe at the same time

        times = Time(np.unique(times.jd), format="jd")
        # Calculate planet positions at all times
        syst_M = []
        for planet in self.planets:
            syst_M.append(planet.mean_anom(times))
        breakpoint()
        E = kt.eccanom_orvara(np.stack(syst_M).value, self.getpattr("e"))

    def propagate_rv(self, times):
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

    def nbody_vectors(self, times, return_r=True, return_v=False):
        """
        Calculate the barycentric position and velocity vectors of all bodies in the system
        at the given times using rebound.
        """
        # Set up rebound simulation
        sim = rebound.Simulation()
        sim.G = const.G.value
        times_jd = times.utc.jd
        sim.t = times_jd[0]
        times = times.datetime64

        # Add the star and planets to the simulation, currently assuming no binary systems
        n_stars = 1
        n_planets = len(self.planets)
        sim = self.add_objects_to_rebound(sim)
        sim.move_to_com()

        data_vars = ["x", "y", "z", "vx", "vy", "vz"]
        coords = {
            "time": times,
            "body_type": ["star", "planet"],
            "body_index": np.arange(max(n_stars, n_planets)),
            "variable": data_vars,
        }
        da = xr.DataArray(
            np.nan, coords=coords, dims=["time", "body_type", "body_index", "variable"]
        )
        da.attrs["units"] = {
            "x": u.m,
            "y": u.m,
            "z": u.m,
            "vx": u.m / u.s,
            "vy": u.m / u.s,
            "vz": u.m / u.s,
        }

        for time_jd, time in tqdm(
            zip(times_jd, times), total=len(times), desc="n-body system propagation"
        ):
            sim.integrate(time_jd)
            for j, p in enumerate(sim.particles):
                body_type = "star" if j < n_stars else "planet"
                body_index = j if j < n_stars else j - n_stars

                da.loc[time, body_type, body_index, "x"] = p.x
                da.loc[time, body_type, body_index, "y"] = p.y
                da.loc[time, body_type, body_index, "z"] = p.z
                da.loc[time, body_type, body_index, "vx"] = p.vx
                da.loc[time, body_type, body_index, "vy"] = p.vy
                da.loc[time, body_type, body_index, "vz"] = p.vz
        return da

    def add_objects_to_rebound(self, sim):
        """
        Method that adds the star and planets to the rebound simulation
        """
        sim.add(
            m=self.star.mass.decompose().value,
            x=self.star._x[0].decompose().value,
            y=self.star._y[0].decompose().value,
            z=self.star._z[0].decompose().value,
            vx=self.star._vx[0].decompose().value,
            vy=self.star._vy[0].decompose().value,
            vz=self.star._vz[0].decompose().value,
        )
        for planet in self.planets:
            sim.add(
                m=planet.mass.decompose().value,
                x=planet._x[0].decompose().value,
                y=planet._y[0].decompose().value,
                z=planet._z[0].decompose().value,
                vx=planet._vx[0].decompose().value,
                vy=planet._vy[0].decompose().value,
                vz=planet._vz[0].decompose().value,
            )
        return sim
