import os
import glob
from typing import Dict

import pandas as pd
import yaml
from pathlib import Path
import ase.io
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import joblib
from tqdm.contrib.concurrent import process_map
from functools import partial

from mof_benchmark import base_dir
from mof_benchmark.analysis.common.utils import (
    get_dft_structure,
    get_reference_rmsd,
    get_stage,
    get_trajectory_rmsd,
)
from mof_benchmark.experiments.scripts.utils import logger
from mof_benchmark.experiments.scripts.utils.utils import get_path


def process_file(file: str, ref, calculator, settings) -> Dict:
    try:
        traj = ase.io.read(file, ":")
    except Exception as e:
        logger.error(f"Error reading file {file}: {e}")
        return {}

    rmsd_exp = get_trajectory_rmsd(traj)

    if ref is None:
        rmsd_dft = [np.nan] * len(rmsd_exp)
        dft_volume = np.nan
        dft_lengths = [np.nan] * 3
        dft_angles = [np.nan] * 3
    else:
        rmsd_dft = [np.nan] * len(rmsd_exp)
        if len(ref) == len(traj[0]):  # if numbers of atoms match
            try:
                rmsd_dft = get_reference_rmsd(traj, ref)
            except Exception as e:
                logger.error(f"Error computing RMSD for file {file}: {e}")
                rmsd_dft = [np.nan] * len(rmsd_exp)
        dft_volume = ref.get_volume()
        dft_lengths = ref.cell.lengths()
        dft_angles = ref.cell.angles()

    traj_volumes = [frame.get_volume() for frame in traj]
    lengths = [frame.cell.lengths() for frame in traj]
    angles = [frame.cell.angles() for frame in traj]
    pot_energy = [frame.get_potential_energy() for frame in traj]

    row = {
        "calculator": calculator,
        "structure": Path(file).stem,
        "settings": str(settings),
        "rmsd_exp": rmsd_exp,
        "rmsd_dft": rmsd_dft,
        "volume": traj_volumes,
        "pot_energy": pot_energy,
        "cell_lengths": lengths,
        "cell_angles": angles,
        "dft_volume": dft_volume,
        "dft_lengths": dft_lengths,
        "dft_angles": dft_angles,
        "step": np.arange(len(rmsd_exp)),
        "stage": get_stage(traj),
    }

    return row


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

    output_file = base_dir / "analysis" / "results" / "optimization_results.parquet"
    if os.path.exists(output_file):
        previous_results_df = pd.read_parquet(output_file)
    else:
        previous_results_df = pd.DataFrame()

    settings = (
        settings
        or base_dir / "experiments" / "scripts" / "optimization" / "optimization.yaml"
    )
    settings_dict = yaml.safe_load(open(settings))

    output_dir = Path(get_path(settings_dict["output_dir"], calculator=calculator))

    files = glob.glob(str(output_dir / "*.traj"))

    if not files or len(files) == 0:
        logger.warning(f"No files found in {output_dir}")
        return

    files = [
        file
        for file in files
        if get_index(settings, file, calculator) not in previous_results_df.index
    ]

    if not files or len(files) == 0:
        logger.warning(f"No new files found in {output_dir}.")
        return

    n_cores = joblib.cpu_count()

    structure_names = [Path(file).stem for file in files]
    dft_structures = [
        get_dft_structure(structure.replace("_primitive", ""))[1]
        for structure in structure_names
    ]

    logger.info(f"Running analysis for {len(files)} files on {n_cores} cores.")
    # results = process_map(
    #     lambda file, ref: process_file(file, ref, calculator, settings),
    #     zip(files, dft_structures),
    #     max_workers=n_cores,
    # )
    results = Parallel(n_jobs=n_cores, timeout=999999)(
        delayed(process_file)(file, ref, calculator, settings)
        for file, ref in tqdm(zip(files, dft_structures), total=len(files))
    )

    results = [result for result in results if result != {}]

    if len(results) == 0:
        logger.error("No results found.")
        return

    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.set_index(["calculator", "structure", "settings"], inplace=True)

    df = pd.concat([previous_results_df, results_df])
    df.to_parquet(output_file)


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
        base_dir / "experiments" / "scripts" / "optimization" / "optimization.yaml",
    ]

    calculator_path = base_dir / "setup"
    calculators = [
        key for key in yaml.safe_load(open(calculator_path / "calculators.yaml")).keys()
    ]

    main(args.calculator or calculators, settings=args.settings or settings)
