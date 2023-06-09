import pickle
import subprocess
from pathlib import Path

import astropy.io.fits as fits
import numpy as np
import pandas as pd
from tqdm import tqdm

from exoverses.base.universe import Universe
from exoverses.exovista.system import ExovistaSystem


def create_universe(universe_params):
    data_path = Path(universe_params["data_path"])
    un = universe_params["universe_number"]
    full_path = f"{data_path}/{un}"
    if not Path(full_path).exists():
        get_data([un], data_path)
    universe = ExovistaUniverse(full_path, cache=True)
    return universe


class ExovistaUniverse(Universe):
    """
    Class for the whole exoVista universe
    """

    def __init__(self, path, cache=False):
        """
        Args:
            path (str or Path):
                Location of all the system files. Should be something like "data/1/"
        """
        self.type = "ExoVista"
        if cache:
            cache_base = Path(".cache", path.split("/")[1])
            if not cache_base.exists():
                cache_base.mkdir(parents=True)
        self.path = path

        # Load all systems
        p = Path(path).glob("*.fits")
        system_files = [x for x in p if x.is_file]
        self.systems = []
        for system_file in tqdm(
            system_files, desc="Loading systems", position=0, leave=False
        ):
            if cache:
                cache_file = Path(cache_base, system_file.stem + ".p")
                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        system = pickle.load(f)
                else:
                    system = ExovistaSystem(system_file)
                    with open(cache_file, "wb") as f:
                        pickle.dump(system, f)
                self.systems.append(system)
            else:
                system = ExovistaSystem(system_file)
                if system is not None:
                    self.systems.append(system)

        # Get star ids
        self.ids = [system.star.id for system in self.systems]
        self.names = [system.star.name for system in self.systems]

        Universe.__init__(self)


def runcmd(cmd, verbose=False):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def get_data(universes=np.arange(1, 13), cache_location="data"):
    """
    This function gets all the exoVista data. It gets the csv file with the universe
    information and puts it in the "data/{universe_number}/target_database.csv".
    Then it goes through every url in the csv file and pulls that fits file into
    "data/{universe_number}/{file}.fits".
    """
    # Iterate over the different universes
    for n in tqdm(universes, position=0, desc="Universe", leave=False):
        universe_url = (
            "https://ckan-files.emac.gsfc.nasa.gov/"
            f"exovista/DEC21/{n}/target_database.csv"
        )
        Path(cache_location, str(n)).mkdir(parents=True, exist_ok=True)
        if not Path(cache_location, str(n), "target_database.csv").exists():
            runcmd(f"wget --directory-prefix=data/{n} {universe_url}", verbose=False)

        df = pd.read_csv(
            Path(cache_location, str(n), "target_database.csv"), low_memory=False
        )
        for i in tqdm(
            np.arange(1, df.shape[0]), position=1, desc="System", leave=False
        ):
            # Get file url
            fit_url = df.at[i, "URL"]

            # Create file path
            file_path = Path(cache_location, str(n), f"{fit_url.split('/')[-1]}")

            # If file doesn't exist then pull it
            if not file_path.exists():
                runcmd(
                    f"wget --directory-prefix=data/{n} {fit_url}",
                    verbose=False,
                )

            # Verify data and repull it if necessary
            pull_failure = True
            while pull_failure:
                pull_failure = check_data(file_path)
                if pull_failure:
                    runcmd(
                        f"wget --directory-prefix=data/{n} {fit_url}",
                        verbose=False,
                    )


def check_data(file):
    """
    This function verifies that all the fits files have the necessary data, sometimes
    they don't pull everything for some reason
    """
    system_info = fits.info(file, output=False)
    with open(file, "rb") as f:
        # read header of first extension
        h = fits.getheader(f, ext=0, memmap=False)
    n_ext = h["N_EXT"]  # get the largest extension
    failure = len(system_info) != n_ext + 1
    if failure:
        # Number of tables doesn't match the number of tables that the header
        # says exists, delete file
        file.unlink()
    return failure
