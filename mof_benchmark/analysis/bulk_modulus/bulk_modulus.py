import os
import glob
from typing import Dict

import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
import joblib
from joblib import Parallel, delayed
from tqdm.contrib.concurrent import process_map
from functools import partial

from mof_benchmark import base_dir  # type: ignore
from mof_benchmark.experiments.scripts.utils import logger
from mof_benchmark.experiments.scripts.utils.utils import get_path


def process_file(file: str, reference, calculator, settings) -> Dict:
    data = np.load(file, allow_pickle=True)

    structure_name = Path(file).stem[:-13]
    b_reference = reference[
        reference["structure"] == structure_name.replace("_primitive", "")
    ]
    if len(b_reference) == 0:
        logger.warning(
            f"No DFT reference found for {structure_name.replace('_primitive', '')}"
        )
        dft_b = None
        dft_vol = None
        dft_eng = None
    else:
        dft_b = float(b_reference["B0_GPa"].iloc[0])
        dft_vol = b_reference["volumes_A3"].iloc[0]
        dft_eng = b_reference["energies_au"].iloc[0]

    row = {
        "calculator": calculator,
        "structure": structure_name,
        "settings": str(settings),
        "B": data["B"].item(),
        "dft_B": dft_b,
        "dft_vol": dft_vol,
        "dft_eng": dft_eng,
        "volumes": (
            data["volumes"].astype(np.float64)
            if data["volumes"] is not None and data["volumes"].ndim > 0
            else None
        ),
        "energies": (
            data["energies"].astype(np.float64)
            if data["energies"] is not None and data["energies"].ndim > 0
            else None
        ),
        "v0": data["v0"].item(),
        "e0": data["e0"].item(),
        "eps": data["eps"].item(),
        "npoints": data["npoints"].item(),
    }

    return row


def get_index(settings, file, calculator):
    settings = str(settings)
    structure = Path(file).stem[:-13]
    return (calculator, structure, settings)


def run_analysis(
    calculator: str,
    settings: str | Path | None = None,
    reference: pd.DataFrame | None = None,
):
    """
    Run bulk modulus analysis.

    Parameters:
    ----------
    calculator : str
        Calculator used to compute the trajectory
    settings : str
        Settings used to compute the trajectory
    """

    output_file = base_dir / "analysis" / "results" / "bulk_modulus_results.parquet"
    if os.path.exists(output_file):
        previous_results_df = pd.read_parquet(output_file)
    else:
        previous_results_df = pd.DataFrame()

    settings = (
        settings
        or base_dir / "experiments" / "scripts" / "bulk_modulus" / "bulk_modulus.yaml"
    )
    settings_dict = yaml.safe_load(open(settings))
    output_dir = Path(get_path(settings_dict["output_dir"], calculator=calculator))
    files = glob.glob(str(output_dir / "*.npz"))

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
    logger.info(f"Running analysis for {len(files)} files on {n_cores} cores.")
    results = Parallel(n_jobs=n_cores, timeout=999999)(
        delayed(process_file)(file, reference, calculator, settings)
        for file in tqdm(files)
    )

    results_df = pd.DataFrame(results)  # type: ignore
    results_df = results_df.set_index(["calculator", "structure", "settings"])

    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.concat([previous_results_df, results_df])
    df.to_parquet(output_file)


def main(calculator: str | list[str], settings: str | list[str]):

    if not isinstance(settings, list):
        settings = [settings]

    if not isinstance(calculator, list):
        calculator = [calculator]

    local_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    dft_results = pd.read_csv(
        local_dir
        / "../dft_data/1st_2nd_final_filter_good_bulk_modulus_calc_info_combined.csv"
    )

    for setting in settings:
        for calc in calculator:
            logger.info(
                f"Running analysis for {calc} with settings {Path(setting).stem}"
            )
            run_analysis(calc, settings=setting, reference=dft_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("calculator", type=str, nargs="?", default=None)
    parser.add_argument("--settings", type=str, default=None)
    args = parser.parse_args()

    settings = [
        base_dir / "experiments" / "scripts" / "bulk_modulus" / "bulk_modulus.yaml",
    ]

    calculator_path = base_dir / "setup"
    calculators = [
        key for key in yaml.safe_load(open(calculator_path / "calculators.yaml")).keys()
    ]

    main(args.calculator or calculators, settings=args.settings or settings)  # type: ignore
