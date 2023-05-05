import json
from pathlib import Path

from astropy.time import Time
from EXOSIMS.util.get_module import get_module_from_specs
from tqdm import tqdm

from exoverses.base.universe import Universe
from exoverses.exosims.system import ExosimsSystem


def create_universe(universe_params):
    script_path = Path(universe_params["script"])
    with open(script_path) as f:
        specs = json.loads(f.read())
    assert "seed" in specs.keys(), (
        "For reproducibility the seed should" " not be randomized by EXOSIMS."
    )

    # Need to use SurveySimulation if we want to have a random seed
    SS = get_module_from_specs(specs, "SurveySimulation")(**specs)
    SU = SS.SimulatedUniverse
    universe_params["missionStart"] = specs["missionStart"]
    # SU = get_module_from_specs(specs, "SimulatedUniverse")(**specs)
    universe = ExosimsUniverse(SU, universe_params)
    return universe


class ExosimsUniverse(Universe):
    """
    Class for the whole EXOSIMS universe
    """

    def __init__(self, SU, params):
        """
        Args:
            path (str or Path):
                Location of all the system files. Should be something like "data/1/"
        """
        self.type = "EXOSIMS"
        self.SU = SU
        self.t0 = Time(params["missionStart"], format="mjd")
        # Load all systems
        sInds = SU.sInds
        if "nsystems" in params.keys():
            nsystems = params["nsystems"]
            sInds = sInds[:nsystems]
        else:
            sInds = SU.sInds
            nsystems = len(sInds)

        if "cache_path" in params.keys():
            self.cache_path = Path(params["cache_path"])

        self.systems = []
        for sInd in tqdm(sInds, desc="Loading systems", position=0, leave=False):
            system = ExosimsSystem(SU, sInd, self.t0)
            self.systems.append(system)

        # Get star ids
        self.ids = [system.star.id for system in self.systems]
        self.names = [system.star.name for system in self.systems]

        Universe.__init__(self)
