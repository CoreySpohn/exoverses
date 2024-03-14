import astropy.units as u
import numpy as np
from astropy.io.fits import getheader
from keplertools import fun as kt

import exoverses.exovista as ev
from exoverses.base.system import System


class ExovistaSystem(System):
    """
    Class for the whole stellar system

    Args:
        infile (Path):
            Path to the exoVista fits file
    """

    def __init__(self, infile, initial_epoc=2000, convert=False, filter=True):
        self.file = infile
        self.name = self.file.name

        # fits file extensions, exoVista hard codes these
        planet_ext = 5
        disk_ext = 2

        # Get the number of planets
        with open(infile, "rb") as f:
            # read header of first extension
            h = getheader(f, ext=0, memmap=False)
        # get the largest extension
        n_ext = h["N_EXT"]
        nplanets = n_ext - 4

        # Create star object
        self.star = ev.star.ExovistaStar(infile)
        self.nstar_wavelengths = len(self.star._wavelengths)
        self.nstar_times = len(self.star._t)
        self.planets = []
        # loop over all planets
        for i in range(nplanets):
            self.planets.append(
                ev.planet.ExovistaPlanet(infile, planet_ext + i, self.star)
            )
        self.disk = ev.disk.ExovistaDisk(infile, disk_ext, self.star)
        self.ndisk_wavelengths = len(self.disk._wavelengths)

        # mas/pixel
        self.pixel_scale = self.star.pixel_scale

        # What the nbody simulation frame is
        self.nbody_frame = "bary-sky"
        self.origin = "ExoVista"

        self.planet_cleanup()
        self.star_cleanup()

        if convert:
            # Use nbody integration to get the positions in the sky-frame
            _t0 = self.planets[0]._t[0].reshape(1)
            bary = self.propagate(_t0, clean=True, ref_frame="bary-sky", prop="nbody")

            # Now we're converting the system object to the "bary" frame from
            # the "bary-sky" frame without a midplane
            self.nbody_frame = "bary"

            # Set the star's position and velocity
            sel = bary.sel(object="star", index=0, ref_frame="bary-sky").squeeze()
            _x, _y, _z = sel.x.item(), sel.y.item(), sel.z.item()
            _vx, _vy, _vz = sel.vx.item(), sel.vy.item(), sel.vz.item()
            self.star._x = u.Quantity(_x * u.m).reshape(1)
            self.star._y = u.Quantity(_y * u.m).reshape(1)
            self.star._z = u.Quantity(_z * u.m).reshape(1)
            self.star._vx = u.Quantity(_vx * u.m / u.s).reshape(1)
            self.star._vy = u.Quantity(_vy * u.m / u.s).reshape(1)
            self.star._vz = u.Quantity(_vz * u.m / u.s).reshape(1)

            # Set the planet's positions, velocities, and orbital angles
            for i, planet in enumerate(self.planets):
                sel = bary.sel(object="planet", index=i, ref_frame="bary-sky").squeeze()
                _x, _y, _z = sel.x.item(), sel.y.item(), sel.z.item()
                _vx, _vy, _vz = sel.vx.item(), sel.vy.item(), sel.vz.item()
                # Assign the new positions and velocities
                planet._x = u.Quantity(_x * u.m).reshape(1)
                planet._y = u.Quantity(_y * u.m).reshape(1)
                planet._z = u.Quantity(_z * u.m).reshape(1)
                planet._vx = u.Quantity(_vx * u.m / u.s).reshape(1)
                planet._vy = u.Quantity(_vy * u.m / u.s).reshape(1)
                planet._vz = u.Quantity(_vz * u.m / u.s).reshape(1)

                # Calculate the orbital elements
                rs = np.array([_x, _y, _z])
                vs = np.array([_vx, _vy, _vz])
                mu = planet.mu.decompose().value.reshape(1)
                _, _, _E, _O, _I, _w, _P, _tau = kt.vec2orbElem(rs, vs, mu)

                # Set the angles
                planet.W = (_O[0] * u.rad).to(u.deg)
                planet.w = (_w[0] * u.rad).to(u.deg)
                planet.inc = (_I[0] * u.rad).to(u.deg)
                _E = _E[0] * u.rad
                planet.M0 = (
                    (_E.value - planet.e * np.sin(_E)) % (2 * np.pi) * u.rad
                ).to(u.deg)
                planet.solve_dependent_params()
                planet.classify_planet()
        if filter:
            earth_ind = self.getpattr("is_earth")
            K_vals = self.getpattr("K")
            max_K = K_vals.max()
            max_earth_K = self.getpattr("K")[np.argwhere(earth_ind).ravel()]
            # Remove all planets with K values larger than the largest K value of the Earth
            # except the larget K value
            to_remove = np.argwhere(
                (K_vals > max_earth_K.max()) & (K_vals != max_K)
            ).ravel()
            self.planets = [
                self.planets[i] for i in range(len(self.planets)) if i not in to_remove
            ]
            self.planet_cleanup()

    def spec_flux_densities(self, wavelengths, times):
        """
        Calculate the spectral flux density of the system at the given times and
        wavelengths
        Args:
            wavelengths (astropy Quantity array):
                Wavelengths to calculate the spectral flux density at
            times (astropy Time array):
                Times to calculate the spectral flux density at
        Returns:
            star_flux_density (astropy Quantity array):
                Spectral flux density of the star at the given times in Jy
                The shape of the cube is (Ntimes, Nwavelengths)
            planet_flux_density (astropy Quantity array):
                Spectral flux density of the planets at the given times in Jy.
                The shape of the cube is (Nplanets, Ntimes, Nwavelengths)
            disk_flux_density (astropy Quantity array):
                Spectral flux density of the disk at the given times in Jy per
                pixel. The shape of the cube is (Ntimes, Nwavelengths, Npix,
                                                 Npix)

        """
        # Calculate the spectral flux density of the star
        star_flux_density = self.star.spec_flux_density(wavelengths, times)

        # Calculate the spectral flux density of the planets
        planet_flux_density = (
            np.zeros((len(self.planets), len(times), len(wavelengths))) * u.Jy
        )
        for i, planet in enumerate(self.planets):
            planet_flux_density[i] += planet.spec_flux_density(wavelengths, times)

        # Calculate the spectral flux density of the disk
        disk_flux_density = self.disk.spec_flux_density(wavelengths, times)
        return star_flux_density, planet_flux_density, disk_flux_density

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
