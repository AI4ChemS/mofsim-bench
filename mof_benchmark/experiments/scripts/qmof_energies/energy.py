from typing import List

import ase
import pandas as pd
import os

from mof_benchmark.experiments.scripts.common.runner import TaskRunner
from mof_benchmark.experiments.scripts.utils import logger
from mof_benchmark import base_dir


class EnergyRunner(TaskRunner):
    def __init__(
        self,
        calculator: str | None = None,
        structure: str | List[str] | None = None,
        settings: str | dict | None = None,
        index: int | None = None,
        **kwargs,
    ):

        files_path = (
            base_dir
            / "experiments"
            / "structures"
            / "qmof_database"
            / "relaxed_structures"
        )
        files = os.listdir(files_path)
        files = [os.path.join(files_path, file) for file in files]

        super().__init__(calculator, files, settings, {}, index, **kwargs)

    def skip(self, **kwargs):
        return False  # if os.path.exists(kwargs.get("structure_name") + ".traj")

    def task(self, atoms: ase.Atoms, settings: dict, **kwargs):

        structure_name = kwargs["structure_name"]
        calculator = kwargs["calculator"]
        output_file = f"{calculator}.parquet"

        logger.info(f"Starting energy calculation for {structure_name}")

        energy = atoms.get_potential_energy()
        logger.info(f"Energy of {structure_name}: {energy}")

        df = pd.DataFrame(
            {
                "structure": [structure_name],
                "energy": [energy],
            }
        )

        if os.path.exists(output_file):
            old_df = pd.read_parquet(output_file)
        else:
            old_df = pd.DataFrame()

        df = pd.concat([old_df, df], ignore_index=True)
        df = df.drop_duplicates(subset=["structure"])
        df.to_parquet(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--calculator", type=str, help="Calculator to use for relaxation."
    )
    parser.add_argument(
        "-s",
        "--structure",
        type=str,
        help="Path to the structure file or shortcut string.",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="Path to the settings file or dictionary with the settings.",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Index of the structure to use. If not provided, all structures found in settings will be used.",
    )
    args = parser.parse_args()

    runner = EnergyRunner(args.calculator, args.structure, args.settings, args.index)
    runner.run()
