import os
import glob
import struct
from typing import Dict

import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import joblib
from tqdm.contrib.concurrent import process_map
from functools import partial

from mof_benchmark import base_dir
from mof_benchmark.experiments.scripts.utils import logger
from mof_benchmark.experiments.scripts.utils.utils import get_path


def process_file(file: str, moosavi_results, calculator, settings) -> Dict:
    data = np.load(file, allow_pickle=True)

    structure_name = Path(file).stem[:-19]
    dft_reference = moosavi_results[
        moosavi_results["cif_name"] == structure_name.replace("_primitive", "")
    ]
    if len(dft_reference) == 0:
        logger.warning(
            f"No DFT reference found for {structure_name.replace('_primitive', '')}"
        )
        dft_cv = None
    else:
        dft_cv = float(dft_reference["cv_300K_JperKperg"].values[0])

    row = {
        "calculator": calculator,
        "structure": structure_name,
        "settings": str(settings),
        "temperatures": data["temperatures"],
        "heat_capacity_mol": data["heat_capacity_mol"],
        "heat_capacity_g": data["heat_capacity_g"],
        "dft_cv": dft_cv,
        "entropy": data["entropy"],
        "free_energy": data["free_energy"],
        "t_min": data["t_min"].item(),
        "t_max": data["t_max"].item(),
        "t_step": data["t_step"].item(),
    }

    return row


def get_index(settings, file, calculator):
    settings = str(settings)
    structure = Path(file).stem[:-19]
    return (calculator, structure, settings)


def run_analysis(calculator: str, settings: str | None = None):
    """
    Run heat capacity analysis.

    Parameters:
    ----------
    calculator : str
        Calculator used
    settings : str
        Settings used
    """

    output_file = base_dir / "analysis" / "results" / "heat_capacity_results.parquet"
    if os.path.exists(output_file):
        previous_results_df = pd.read_parquet(output_file)
    else:
        previous_results_df = pd.DataFrame()

    settings = (
        settings
        or base_dir / "experiments" / "scripts" / "heat_capacity" / "heat_capacity.yaml"
    )
    settings_dict = yaml.safe_load(open(settings))

    local_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    output_dir = Path(get_path(settings_dict["output_dir"], calculator=calculator))

    files = glob.glob(str(output_dir / "*.npz"))

    if not files or len(files) == 0:
        logger.warning(f"No files found in {output_dir}")
        return

    moosavi_results = pd.read_csv(local_dir / "collect_cv_300k.csv")

    files = [
        file
        for file in files
        if get_index(settings, file, calculator) not in previous_results_df.index
    ]

    if not files or len(files) == 0:
        logger.warning(f"No new files found in {output_dir}.")
        return

    n_cores = joblib.cpu_count()
    logger.info(f"Running analysis for {len(files)} files on {n_cores} cores.")
    results = Parallel(n_jobs=n_cores, timeout=999999)(
        delayed(process_file)(file, moosavi_results, calculator, settings)
        for file in tqdm(files)
    )

    results_df = pd.DataFrame(results)
    results_df = results_df.set_index(["calculator", "structure", "settings"])

    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

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
        base_dir / "experiments" / "scripts" / "heat_capacity" / "heat_capacity.yaml",
    ]

    calculator_path = base_dir / "setup"
    calculators = [
        key for key in yaml.safe_load(open(calculator_path / "calculators.yaml")).keys()
    ]

    main(args.calculator or calculators, settings=args.settings or settings)
