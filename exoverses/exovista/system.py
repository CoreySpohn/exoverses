from astropy.io.fits import getheader

import exoverses.exovista as ev
from exoverses.base.system import System


class ExovistaSystem(System):
    """
    Class for the whole stellar system
    """

    def __init__(self, infile):
        self.file = infile

        # fits file extensions, exoVista hard codes these
        planet_ext = 4

        # Get the number of planets
        with open(infile, "rb") as f:
            # read header of first extension
            h = getheader(f, ext=0, memmap=False)
        n_ext = h["N_EXT"]  # get the largest extension
        nplanets = n_ext - 3

        # Create star object
        self.star = ev.star.ExovistaStar(infile)
        self.planets = []
        # loop over all planets
        for i in range(nplanets):
            self.planets.append(
                ev.planet.ExovistaPlanet(infile, planet_ext + i, self.star)
            )

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
