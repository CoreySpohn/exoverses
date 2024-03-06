import copy

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time

from exoverses.base.system import System
from exoverses.fit.planet import FitPlanet


class FitSystem(System):
    def __init__(self, search_obj, true_system):
        """
        This function creates a rvtools system object with the planets found
        during the fitting process. It also calculates the planet that the
        fit is most likely describing.
        """
        search_params = search_obj.post.params
        self.star = copy.deepcopy(true_system.star)
        self.planets = []
        for nplan in range(1, search_obj.num_planets + 1):
            # Create dictionary to initialize planet with
            plan_dict = {}
            plan_dict["T"] = search_params[f"per{nplan}"].value * u.d
            plan_dict["T_c"] = Time(search_params[f"tc{nplan}"].value, format="jd")
            plan_dict["secosw"] = search_params[f"secosw{nplan}"].value
            plan_dict["sesinw"] = search_params[f"sesinw{nplan}"].value
            plan_dict["K"] = search_params[f"k{nplan}"].value * u.m / u.s

            # Initialize the planet object
            planet = FitPlanet(plan_dict, true_system)
            self.planets.append(planet)
        self.pInds = np.arange(0, search_obj.num_planets)
        self.cleanup()
        self.true_system = true_system

    def get_p_df(self):
        patts = [
            "K",
            "T",
            "secosw",
            "sesinw",
            "T_c",
            "a",
            "e",
            "w",
            "M0",
            "t0",
            "msini",
            "best_match",
            "best_rms",
        ]
        p_df = pd.DataFrame()
        for att in patts:
            pattr = self.getpattr(att)
            if type(pattr) == u.Quantity:
                p_df[att] = pattr.value
            else:
                p_df[att] = pattr

        return p_df
