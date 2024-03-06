import numpy as np

from exoverses.base.system import System
from exoverses.exosims.planet import ExosimsPlanet
from exoverses.exosims.star import ExosimsStar


class ExosimsSystem(System):
    """
    Class for the whole stellar system
    """

    def __init__(self, SU, sInd, t0):
        # Create star object
        self.star = ExosimsStar(SU, sInd)
        self.planets = []
        self.pInds = np.where(SU.plan2star == sInd)[0]
        # loop over all planets
        for pInd in self.pInds:
            self.planets.append(ExosimsPlanet(SU, self.star, pInd, t0))

        self.cleanup()
        self.origin = "EXOSIMS"
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
