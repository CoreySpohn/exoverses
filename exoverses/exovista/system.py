import astropy.units as u
import numpy as np
from astropy.io.fits import getheader

import exoverses.exovista as ev
from exoverses.base.system import System


class ExovistaSystem(System):
    """
    Class for the whole stellar system

    Args:
        infile (Path):
            Path to the exoVista fits file
    """

    def __init__(self, infile, initial_epoc=2000):
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

        # self.ndisk_times = len(self.disk.ev_t)

        # self.cleanup()

        # Set up rebound simulation
        # self.sim = rebound.Simulation()
        # self.sim.G = const.G.value
        # self.sim.add(
        #     m=self.star.mass.decompose().value,
        #     x=self.star._x[0].decompose().value,
        #     y=self.star._y[0].decompose().value,
        #     z=self.star._z[0].decompose().value,
        #     vx=self.star._vx[0].decompose().value,
        #     vy=self.star._vy[0].decompose().value,
        #     vz=self.star._vz[0].decompose().value,
        # )
        # for planet in self.planets:
        #     self.sim.add(
        #         m=planet.mass.decompose().value,
        #         x=planet._x[0].decompose().value,
        #         y=planet._y[0].decompose().value,
        #         z=planet._z[0].decompose().value,
        #         vx=planet._vx[0].decompose().value,
        #         vy=planet._vy[0].decompose().value,
        #         vz=planet._vz[0].decompose().value,
        #     )
        # self.sim.move_to_com()

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
