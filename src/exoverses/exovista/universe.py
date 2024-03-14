import multiprocessing
import os
import subprocess
from pathlib import Path

import astropy.io.fits as fits
import dill
import numpy as np
import pandas as pd
from ExoVista import (
    Settings,
    generate_disks,
    generate_planets,
    generate_scene,
    load_stars,
    read_solarsystem,
)
from tqdm import tqdm

from exoverses.base.universe import Universe
from exoverses.exovista.system import ExovistaSystem


def create_universe(universe_params, workers=10):
    data_path = Path(universe_params["data_path"])
    targetlist = universe_params["target_list"]
    convert = universe_params.get("convert")
    # system_path is used to generate a specific system
    system_path = universe_params.get("system_path")
    full_path = f"{data_path}"
    has_system_path = system_path is not None
    if has_system_path:
        settings = Settings.Settings(timemax=10.0, ncomponents=2, output_dir=full_path)
        s, p, a, d, c, new_settings = read_solarsystem.read_solarsystem(
            settings, system_file=system_path
        )
        generate_scene.generate_scene(s, p, d, a, c, new_settings)
    else:
        generate_systems(targetlist, full_path, workers=workers)

    universe = ExovistaUniverse(full_path, targetlist, convert=convert, cache=True)
    return universe


class ExovistaUniverse(Universe):
    """
    Class for the whole exoVista universe
    """

    def __init__(self, path, target_list, convert=False, cache=False):
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
        hip_inds = pd.read_csv(target_list)
        if "HIP" not in hip_inds.columns:
            hip_inds = hip_inds.rename(columns={"name": "HIP"})
        hip_inds["HIP"] = hip_inds["HIP"].str.replace("HIP ", "").astype(int)
        relevant_files = []
        for file in system_files:
            file_hip_ind = int(file.stem.split("-")[1].split("_")[1])
            if file_hip_ind in hip_inds["HIP"].values:
                relevant_files.append(file)
        self.systems = []
        for system_file in tqdm(
            relevant_files, desc="Loading systems", position=0, leave=False
        ):
            if cache:
                cache_file = Path(cache_base, "exoverses", system_file.stem + ".p")
                if not cache_file.parent.exists():
                    cache_file.parent.mkdir(parents=True)
                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        system = dill.load(f)
                else:
                    system = ExovistaSystem(system_file, convert=convert)
                    with open(cache_file, "wb") as f:
                        dill.dump(system, f)
                self.systems.append(system)
            else:
                system = ExovistaSystem(system_file)
                if system is not None:
                    self.systems.append(system)

        # Get star ids
        self.ids = [system.star.id for system in self.systems]
        self.names = [system.star.name for system in self.systems]

        super().__init__()


def generate_systems(targetlist, path, workers=12):
    settings = Settings.Settings(timemax=10.0, output_dir=path)

    stars, nexozodis = load_stars.load_stars(targetlist, from_master=True)
    print("\n{0:d} stars in model ranges.".format(len(stars)))

    planets, albedos = generate_planets.generate_planets(
        stars, settings, force_earth=True
    )
    disks, compositions = generate_disks.generate_disks(
        stars, planets, settings, nexozodis=nexozodis
    )
    print("Generating scenes. (This may take a while.)")

    cores = min(workers, os.cpu_count())
    cores = min(cores, len(stars))
    percore = int(np.ceil(len(stars) / cores))
    if percore > 1 and percore * (cores - 1) >= len(stars):
        percore -= 1

    print(f"Using {cores} cores")
    pool = multiprocessing.Pool(cores)

    inputs = []
    for i in range(0, cores):
        imin = i * percore
        imax = (i + 1) * percore
        inputs.append(
            [
                stars.iloc[imin:imax],
                planets[imin:imax],
                disks[imin:imax],
                albedos[imin:imax],
                compositions[imin:imax],
                settings,
            ]
        )

    pool.starmap(generate_scene.generate_scene, [inputs[j][:] for j in range(0, cores)])
    pool.close()
    pool.join()


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
