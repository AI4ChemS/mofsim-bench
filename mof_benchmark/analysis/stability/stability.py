import os
import glob
from typing import Dict, Tuple

import pandas as pd
import yaml
from pathlib import Path
import ase.io
from tqdm import tqdm
import numpy as np
from ase.data import chemical_symbols
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import CrystalNN
from joblib import Parallel, delayed
import joblib
from tqdm.contrib.concurrent import process_map
from functools import partial

from mof_benchmark import base_dir
from mof_benchmark.analysis.common.utils import (
    get_coordination_numbers,
    get_stage,
    get_trajectory_rmsd,
    is_metal,
)
from mof_benchmark.experiments.scripts.utils import logger
from mof_benchmark.experiments.scripts.utils.utils import get_path


def squeeze_list(lst: list) -> list:
    if isinstance(lst[0], list) or isinstance(lst[0], np.ndarray):
        return [item for sublist in lst for item in sublist]
    return lst


def process_file(file: str, calculator: str, settings: str) -> Dict:

    if not os.path.exists(file):
        logger.error(f"File {file} not found")
        return {}

    try:
        traj = ase.io.read(file, ":")

        # Compute trajectory properties
        rmsd_exp = get_trajectory_rmsd(traj)
        traj_volumes = [frame.get_volume() for frame in traj]
        traj_volumes_rel = np.array(traj_volumes) / traj_volumes[0]
        lengths = [frame.cell.lengths() for frame in traj]
        angles = [frame.cell.angles() for frame in traj]
        temperatures = [frame.get_temperature() for frame in traj]
        steps = [frame.info.get("step", -1) for frame in traj]

        stages = get_stage(traj)

        potential_energy = squeeze_list(
            [frame.get_potential_energy() for frame in traj]
        )  # posegnn returns energy as single element ndarray
        kinetic_energy = [frame.get_kinetic_energy() for frame in traj]

        center_of_mass = [frame.get_center_of_mass() for frame in traj]
        center_of_mass_drift = np.linalg.norm(
            np.array(center_of_mass) - np.array(center_of_mass[0]), axis=1
        )

        # Compute coordination numbers at the beginning and end of the trajectory
        numbers = traj[0].get_atomic_numbers()
        where_metal = is_metal(numbers)
        metal_numbers = numbers[where_metal]
        nn = CrystalNN(x_diff_weight=1.5, search_cutoff=4.5)
        structure = AseAtomsAdaptor.get_structure(traj[0])
        coordination_numbers_initial = np.asarray(
            [nn.get_cn(structure, int(i)) for i in where_metal.nonzero()[0]]
        )
        structure = AseAtomsAdaptor.get_structure(traj[-1])
        coordination_numbers_final = np.asarray(
            [nn.get_cn(structure, int(i)) for i in where_metal.nonzero()[0]]
        )

        coordination_df = pd.DataFrame(
            {
                "numbers": metal_numbers,
                "initial_coordination": coordination_numbers_initial,
                "final_coordination": coordination_numbers_final,
            }
        )
        mean_coordination_numbers = (
            coordination_df.groupby("numbers").mean().reset_index()
        )
        symbols = [
            chemical_symbols[number] for number in mean_coordination_numbers["numbers"]
        ]

        row = {
            "calculator": calculator,
            "structure": Path(file).stem,
            "settings": str(settings),
            "rmsd_exp": rmsd_exp,
            "volume": traj_volumes,
            "volume_rel": traj_volumes_rel,
            "lengths": lengths,
            "angles": angles,
            "step": steps,
            "potential_energy": potential_energy,
            "kinetic_energy": kinetic_energy,
            "center_of_mass_drift": center_of_mass_drift,
            "temperature": temperatures,
            "stage": stages,
            "initial_coordination": mean_coordination_numbers[
                "initial_coordination"
            ].to_numpy(),
            "final_coordination": mean_coordination_numbers[
                "final_coordination"
            ].to_numpy(),
            "symbol": symbols,
        }

        return row

    except Exception as e:
        logger.error(f"Error in {file}: {e}")
        return {}


def get_index(settings, file, calculator):
    settings = str(settings)
    structure = Path(file).stem
    return (calculator, structure, settings)


def run_analysis(calculator: str, settings: str | None = None):
    """
    Run optimization analysis.

    Parameters:
    ----------
    calculator : str
        Calculator used to compute the trajectory
    settings : str
        Settings used to compute the trajectory
    """
    output_file = base_dir / "analysis" / "results" / "stability_results.parquet"
    if os.path.exists(output_file):
        previous_results_df = pd.read_parquet(output_file)
    else:
        previous_results_df = pd.DataFrame()

    settings = (
        settings
        or base_dir / "experiments" / "scripts" / "stability" / "stability_ambient.yaml"
    )
    settings_dict = yaml.safe_load(open(settings))

    output_dir = Path(get_path(settings_dict["output_dir"], calculator=calculator))

    files = glob.glob(str(output_dir / "*.traj"))

    if len(files) == 0:
        logger.error(f"No trajectory files found in {output_dir}")
        return

    files = [
        file
        for file in files
        if get_index(settings, file, calculator) not in previous_results_df.index
    ]

    if len(files) == 0:
        logger.error(f"No new files found in {output_dir}")
        return

    n_cores = joblib.cpu_count()
    logger.info(f"Using {n_cores} cores for processing.")
    # results = process_map(
    #     partial(process_file, calculator=calculator, settings=settings),
    #     files,
    #     max_workers=n_cores,
    # )
    results = Parallel(n_jobs=n_cores, timeout=999999)(
        delayed(process_file)(file, calculator, settings) for file in tqdm(files)
    )
    # results = [process_file(file, calculator, settings) for file in tqdm(files)]

    # Save results

    results = [result for result in results if result != {}]

    if len(results) > 0:
        logger.info("Saving results...")
        if not os.path.exists(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        results_df = pd.DataFrame(results)
        results_df = results_df.set_index(["calculator", "structure", "settings"])

        df = pd.concat([results_df, previous_results_df])
        df.to_parquet(output_file)

        logger.info("Results saved.")


def main(calculator: str | list[str], settings: str | list[str] | None = None):

    if not isinstance(settings, list):
        settings = [settings]

    if not isinstance(calculator, list):
        calculator = [calculator]

    for setting in settings:
        for calc in calculator:
            logger.info(
                f"Running analysis for {calc} with settings {Path(setting).stem}"
            )
            run_analysis(calc, settings=setting)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("calculator", type=str, nargs="?", default=None)
    parser.add_argument("--settings", type=str, default=None)
    args = parser.parse_args()

    settings = [
        base_dir / "experiments" / "scripts" / "stability" / "stability_prod_mtk.yaml",
        base_dir
        / "experiments"
        / "scripts"
        / "stability"
        / "stability_prod_copper_mtk.yaml",
        base_dir
        / "experiments"
        / "scripts"
        / "stability"
        / "stability_prod_temp_mtk.yaml",
    ]

    calculator_path = base_dir / "setup"
    calculators = [
        key for key in yaml.safe_load(open(calculator_path / "calculators.yaml")).keys()
    ]

    main(args.calculator or calculators, settings=args.settings or settings)
