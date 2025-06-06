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

from mof_benchmark import base_dir
from mof_benchmark.experiments.scripts.utils import logger
from mof_benchmark.experiments.scripts.utils.utils import get_path


def mae(x, y):
    """
    Mean Absolute Error
    """
    return np.mean(np.abs(x - y))


def process_file(file: str, calculator, ref_df) -> Dict:
    interaction_energy_df = pd.read_parquet(file)
    if interaction_energy_df.empty:
        logger.warning(f"File {file} is empty. Skipping.")
        return {}

    interaction_energy_df.set_index(["structure"], inplace=True)

    energy_co2 = interaction_energy_df.loc["CO2"]["potential_energy"]
    energy_h2o = interaction_energy_df.loc["H2O"]["potential_energy"]

    rows = []
    for structure in interaction_energy_df.index:
        if (
            structure in ["CO2", "H2O"] or len(structure.split("_")) <= 2
        ):  # skip if molecule or framework only
            continue

        energy_total = interaction_energy_df.loc[structure]["potential_energy"]
        energy_mof = interaction_energy_df.loc["_".join(structure.split("_")[:2])][
            "potential_energy"
        ]
        gas_type = structure.split("_")[1]

        if gas_type == "CO2":
            energy_gas = energy_co2
        elif gas_type == "H2O":
            energy_gas = energy_h2o

        interaction_energy = energy_total - energy_mof - energy_gas  # type: ignore
        DFT_interaction_energy = (
            ref_df.loc[structure]["DFT_E_total"]
            - ref_df.loc[structure]["DFT_E_mof"]
            - ref_df.loc[structure]["DFT_E_gas"]
        )
        interaction_energy_mae = mae(interaction_energy, DFT_interaction_energy)
        force_mae = mae(
            np.stack(interaction_energy_df.loc[structure]["forces"]),
            np.stack(ref_df.loc[structure]["forces"]),
        )
        row = {
            "calculator": calculator,
            "structure": structure,
            "energy_total": energy_total,
            "energy_mof": energy_mof,
            "energy_gas": energy_gas,  # type: ignore
            "type": structure.split("_")[2],
            "DFT_E_total": ref_df.loc[structure]["DFT_E_total"],
            "DFT_E_mof": ref_df.loc[structure]["DFT_E_mof"],
            "DFT_E_gas": ref_df.loc[structure]["DFT_E_gas"],
            "interaction_energy": interaction_energy,
            "DFT_interaction_energy": DFT_interaction_energy,
            "interaction_energy_mae": interaction_energy_mae,
            "force_mae": force_mae,
            "n_atoms": ref_df.loc[structure]["n_atoms"],
        }

        rows.append(row)

    return rows


def run_analysis(
    calculator: str, settings: str | None = None, ref_df: pd.DataFrame | None = None
):
    """
    Run optimization analysis.

    Parameters:
    ----------
    calculator : str
        Calculator used to compute the trajectory
    settings : str
        Settings used to compute the trajectory
    """

    output_file = (
        base_dir / "analysis" / "results" / "interaction_energy_results.parquet"
    )
    if os.path.exists(output_file):
        previous_results_df = pd.read_parquet(output_file)
    else:
        previous_results_df = pd.DataFrame()

    settings_dict = yaml.safe_load(open(settings))

    output_dir = Path(get_path(settings_dict["output_dir"], calculator=calculator))

    files = glob.glob(str(output_dir / "*.parquet"))

    if not files or len(files) == 0:
        logger.warning(f"No files found in {output_dir}")
        return

    n_cores = joblib.cpu_count()

    logger.info(f"Running analysis for {len(files)} files on {n_cores} cores.")
    results = Parallel(n_jobs=n_cores, timeout=999999)(
        delayed(process_file)(file, calculator, ref_df)
        for file in tqdm(files, total=len(files))
    )

    if len(results) == 0:
        logger.error("No results found.")
        return

    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = sum(results, [])
    results_df = pd.DataFrame(results)
    results_df.set_index(["calculator", "structure"], inplace=True)

    df = pd.concat([previous_results_df, results_df])
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(output_file)


def main(calculator: str | list[str], settings: str | list[str] | None = None):

    if not isinstance(settings, list):
        settings = [settings]

    if not isinstance(calculator, list):
        calculator = [calculator]

    dft_reference_xyz = ase.io.read("test.xyz", ":")
    references = []
    for atoms in dft_reference_xyz:
        references.append(
            (
                atoms.info["name"],
                atoms.info["DFT_E_total"],
                atoms.info["DFT_E_mof"],
                atoms.info["DFT_E_gas"],
                atoms.arrays["REF_forces"].tolist(),
                len(atoms),
            )
        )
    ref_df = pd.DataFrame(
        references,
        columns=[
            "structure",
            "DFT_E_total",
            "DFT_E_mof",
            "DFT_E_gas",
            "forces",
            "n_atoms",
        ],
    )
    ref_df.set_index("structure", inplace=True)

    for setting in settings:
        for calc in calculator:
            logger.info(
                f"Running analysis for {calc} with settings {Path(setting).stem}"
            )
            run_analysis(calc, setting, ref_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("calculator", type=str, nargs="?", default=None)
    parser.add_argument("--settings", type=str, default=None)
    args = parser.parse_args()

    settings = [
        base_dir
        / "experiments"
        / "scripts"
        / "interaction_energy"
        / "interaction_energy.yaml",
    ]

    calculator_path = base_dir / "setup"
    calculators = [
        key for key in yaml.safe_load(open(calculator_path / "calculators.yaml")).keys()
    ]

    main(args.calculator or calculators, settings=args.settings or settings)
