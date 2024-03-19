import astropy.constants as const
import astropy.units as u
import pandas as pd
from importlib import resources


class Star:
    """
    The star of a system
    """

    def __init__(self, star_dict):
        self.spectral_type = star_dict["spectral_type"]
        self.dist = star_dict["dist"]
        self.name = star_dict["name"]
        self.mass = star_dict["mass"]

    def __repr__(self):
        return f"{type(self).__name__} object\n{self.name}"

    def calc_jitter_terms(self):
        # Granulation noise scaling from Luhn 2020
        self.sigma_gran = (
            1
            * u.m
            / u.s
            * (self.luminosity / const.L_sun) ** 0.5
            * (self.mass / const.M_sun) ** -1
            * (self.effective_temperature / (5777 * u.K)) ** -0.5
        ).decompose()

        # Magnetic activity noise scaling relation from Gupta 2021
        with resources.open_text("exoverses.data", "mamajek_rhk.csv") as f:
            rhk_df = pd.read_csv(f)
        if self.name in rhk_df["HIP"].values:
            logrhk = rhk_df.loc[rhk_df["HIP"] == self.name]["logR'HK"].item()
            logsigma_mag = 1.66 * logrhk + 8.39
            self.sigma_mag = 10**logsigma_mag * u.m / u.s
        else:
            self.sigma_mag = None
