import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
import rebound
import xarray as xr
from astropy.time import Time
from keplertools import fun as kt
from tqdm import tqdm

import exoverses.util.misc as misc


class System:
    """
    Class for a single system. Must have a star and a list of planets.
    """

    def __init__(self, star=None, planets=None, disk=None) -> None:
        self.star = star
        self.planets = planets
        self.disk = disk
        if self.planets is not None:
            self.planet_cleanup()
        if self.star is not None:
            self.star_cleanup()

        self.origin = "Base"
        self.nbody_frame = "bary"

    def __repr__(self):
        return (
            f"{self.star.name}\tdist:{self.star.dist}\t"
            f"Type:{self.star.spectral_type}\n\n"
            f"Planets:\n{self.get_p_df()}"
        )

    def planet_cleanup(self):
        self.pInds = np.arange(len(self.planets))
        # Sort the planets in the system by semi-major axis
        a_vals = [planet.a.value for planet in self.planets]
        self.planets = np.array(self.planets)[np.argsort(a_vals)].tolist()
        self.pInds = self.pInds[np.argsort(a_vals)]

    def star_cleanup(self):
        if hasattr(self.star, "midplane_I"):
            self.midplane_I = self.star.midplane_I
        else:
            self.midplane_I = 0 * u.rad

        if hasattr(self.star, "midplane_PA"):
            self.midplane_PA = self.star.midplane_PA
        else:
            self.midplane_PA = 0 * u.rad

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

    def propagate_rv(self, times):
        """
        Propagates system at all times given and returns the radial velocity

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

        return df

    def create_dataset(self, times):
        """
        Create an xarray Dataset for the system's motion
        """
        timesd64 = times.datetime64
        n_stars = 1
        n_planets = len(self.planets)
        state_vars = ["x", "y", "z", "vx", "vy", "vz"]
        coords = {
            "time": timesd64,
            "object": ["star", "planet", "disk", "fit"],
            "index": np.arange(max(n_stars, n_planets)),
            "ref_frame": ["bary", "helio", "bary-sky", "helio-sky"],
            "prop": ["kepler", "nbody"],
        }

        data_vars = {
            var: (
                ["time", "object", "index", "ref_frame", "prop"],
                np.nan * np.ones((len(timesd64), 4, max(n_stars, n_planets), 4, 2)),
            )
            for var in state_vars
        }

        ds = xr.Dataset(data_vars, coords=coords)

        # Add units information
        for var in state_vars:
            ds[var].attrs["unit"] = u.m if var in ["x", "y", "z"] else u.m / u.s
        return ds

    def prop_kepler(self, times, ds):
        """
        Calculate the barycentric position and velocity vectors of the planets
        in the system at the given times.
        """
        for i, planet in enumerate(self.planets):
            # Calculate the position and velocity vectors
            r, v = planet.calc_vectors(times, return_v=True)
            for j, coord in enumerate(["x", "y", "z"]):
                ds[coord].loc[times.datetime64, "planet", i, "bary", "kepler"] = (
                    r[j].to(u.m).value
                )
            for j, coord in enumerate(["vx", "vy", "vz"]):
                ds[coord].loc[times.datetime64, "planet", i, "bary", "kepler"] = (
                    v[j].to(u.m / u.s).value
                )
        return ds

    def prop_nbody(self, times, ds):
        """
        Calculate the barycentric position and velocity vectors of all bodies
        in the system at the given times using rebound.
        """
        # Set up rebound simulation
        sim = rebound.Simulation()
        sim.G = const.G.decompose().value
        t0 = self.star._t[0]
        times_sec = (times - t0).sec
        times64 = times.datetime64

        # Add the star and planets to the simulation, currently assuming no
        # binary systems
        sim = self.add_objects_to_rebound(sim)
        sim.move_to_com()

        n_stars = 1
        for time_sec, time64 in tqdm(
            zip(times_sec, times64),
            total=len(times),
            desc="n-body system propagation",
            delay=0.5,
        ):
            sim.integrate(time_sec)
            for j, p in enumerate(sim.particles):
                object = "star" if j < n_stars else "planet"
                index = j if j < n_stars else j - n_stars

                for coord, value in zip(["x", "y", "z"], [p.x, p.y, p.z]):
                    ds[coord].loc[time64, object, index, self.nbody_frame, "nbody"] = (
                        value
                    )
                for coord, value in zip(["vx", "vy", "vz"], [p.vx, p.vy, p.vz]):
                    ds[coord].loc[time64, object, index, self.nbody_frame, "nbody"] = (
                        value
                    )
        return ds

    def add_heliocentric_motion(self, ds, prop="kepler"):
        """
        Subtract the star's motion from the system's motion to get the
        heliocentric motion of the planets
        """
        bary_frame = "bary"
        helio_frame = "helio"
        star_bary_r = (
            ds[["x", "y", "z"]]
            .sel(object="star", index=0, ref_frame=bary_frame, prop=prop)
            .to_array()
            .data
        )
        star_bary_v = (
            ds[["vx", "vy", "vz"]]
            .sel(object="star", index=0, ref_frame=bary_frame, prop=prop)
            .to_array()
            .data
        )

        assert np.all(~np.isnan(star_bary_r)), (
            "Star position vector has NaN. Cannot compute heliocentric frame."
            " Probably needs N-body propagation."
        )

        for object in ["star", "planet"]:
            for i in range(len(ds.index)):
                # Rotate the position vectors
                object_bary_r = (
                    ds[["x", "y", "z"]]
                    .sel(object=object, index=i, ref_frame=bary_frame, prop=prop)
                    .to_array()
                    .data
                )
                object_helio_r = object_bary_r - star_bary_r
                ds["x"].loc[:, object, i, helio_frame, prop] = object_helio_r[0]
                ds["y"].loc[:, object, i, helio_frame, prop] = object_helio_r[1]
                ds["z"].loc[:, object, i, helio_frame, prop] = object_helio_r[2]

                # Rotate the velocity vectors
                object_bary_v = (
                    ds[["vx", "vy", "vz"]]
                    .sel(object=object, index=i, ref_frame=bary_frame, prop=prop)
                    .to_array()
                    .data
                )
                object_helio_v = object_bary_v - star_bary_v
                ds["vx"].loc[:, object, i, helio_frame, prop] = object_helio_v[0]
                ds["vy"].loc[:, object, i, helio_frame, prop] = object_helio_v[1]
                ds["vz"].loc[:, object, i, helio_frame, prop] = object_helio_v[2]
        return ds

    def propagate(
        self,
        times,
        ds=None,
        prop="kepler",
        ref_frame="bary",
        convention="exovista",
        clean=False,
    ):
        """
        Wrapper to handle the various propagation methods
        """
        scalar_time = times.isscalar
        if scalar_time:
            times = Time([times])

        if ds is None:
            ds = self.create_dataset(times)
        # Get the barycentric motion
        if prop == "kepler":
            ds = self.prop_kepler(times, ds)
        elif prop == "nbody":
            ds = self.prop_nbody(times, ds)

        # Convert to the desired frame
        ds = self.convert_to_frame(
            ds, ref_frame=ref_frame, convention=convention, prop=prop
        )

        if scalar_time:
            ds = ds.isel(time=0)

        if clean:
            # Check for coordinate values that have all nan values
            # and drop them to save memory space
            ds = misc.drop_nan_coord_values(ds)
        return ds

    def convert_to_frame(
        self, ds, ref_frame="bary", convention="exovista", prop="kepler"
    ):
        if prop == "nbody" and self.nbody_frame == "bary-sky":
            # Add the base frame, local ecliptic barycentric
            ds = self.rotate_to_local_ecliptic_coords(ds, ref_frame="bary", prop=prop)

        if ref_frame in ["helio", "helio-sky"]:
            # Add the heliocentric motion
            ds = self.add_heliocentric_motion(ds, prop=prop)

        if ref_frame in ["bary-sky", "helio-sky"]:
            ds = self.rotate_to_sky_coords(
                ds, ref_frame=ref_frame, convention=convention, prop=prop
            )

        return ds

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

    def rotate_to_sky_coords(
        self, ds, ref_frame="bary-sky", convention="exovista", prop="kepler"
    ):
        """
        Rotate the state vectors to the sky coordinates from barycentric
        coordinates
        """
        if ref_frame == "bary-sky":
            base_frame = "bary"
        elif ref_frame == "helio-sky":
            base_frame = "helio"
        for object in ["star", "planet"]:
            for i in range(len(ds.index)):
                # Rotate the position vectors
                base_r = (
                    ds[["x", "y", "z"]]
                    .sel(object=object, index=i, ref_frame=base_frame, prop=prop)
                    .to_array()
                    .data.T
                )
                sky_r = misc.gen_rotate_to_sky_coords(
                    base_r,
                    self.midplane_I,
                    self.midplane_PA,
                    convention=convention,
                ).T
                ds["x"].loc[:, object, i, ref_frame, prop] = sky_r[0]
                ds["y"].loc[:, object, i, ref_frame, prop] = sky_r[1]
                ds["z"].loc[:, object, i, ref_frame, prop] = sky_r[2]

                # Rotate the velocity vectors
                base_v = (
                    ds[["vx", "vy", "vz"]]
                    .sel(object=object, index=i, ref_frame=base_frame, prop=prop)
                    .to_array()
                    .data.T
                )
                sky_v = misc.gen_rotate_to_sky_coords(
                    base_v,
                    self.midplane_I,
                    self.midplane_PA,
                    convention=convention,
                ).T
                ds["vx"].loc[:, object, i, ref_frame, prop] = sky_v[0]
                ds["vy"].loc[:, object, i, ref_frame, prop] = sky_v[1]
                ds["vz"].loc[:, object, i, ref_frame, prop] = sky_v[2]
        return ds

    def rotate_to_local_ecliptic_coords(
        self, ds, ref_frame="bary", convention="exovista", prop="kepler"
    ):
        """
        Rotate the state vectors to the sky coordinates from barycentric
        coordinates
        """
        if ref_frame == "bary":
            base_frame = "bary-sky"
        elif ref_frame == "helio":
            base_frame = "helio-sky"
        for object in ["star", "planet"]:
            for i in range(len(ds.index)):
                # Rotate the position vectors
                base_r = (
                    ds[["x", "y", "z"]]
                    .sel(object=object, index=i, ref_frame=base_frame, prop=prop)
                    .to_array()
                    .data.T
                )
                sky_r = misc.gen_rotate_to_local_ecliptic_coords(
                    base_r,
                    self.midplane_I,
                    self.midplane_PA,
                    convention=convention,
                ).T
                ds["x"].loc[:, object, i, ref_frame, prop] = sky_r[0]
                ds["y"].loc[:, object, i, ref_frame, prop] = sky_r[1]
                ds["z"].loc[:, object, i, ref_frame, prop] = sky_r[2]

                # Rotate the velocity vectors
                base_v = (
                    ds[["vx", "vy", "vz"]]
                    .sel(object=object, index=i, ref_frame=base_frame, prop=prop)
                    .to_array()
                    .data.T
                )
                sky_v = misc.gen_rotate_to_local_ecliptic_coords(
                    base_v,
                    self.midplane_I,
                    self.midplane_PA,
                    convention=convention,
                ).T
                ds["vx"].loc[:, object, i, ref_frame, prop] = sky_v[0]
                ds["vy"].loc[:, object, i, ref_frame, prop] = sky_v[1]
                ds["vz"].loc[:, object, i, ref_frame, prop] = sky_v[2]
        return ds
