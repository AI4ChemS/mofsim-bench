from typing import List
import os

import ase
import pandas as pd

from mof_benchmark.experiments.scripts.common.runner import TaskRunner
from mof_benchmark.experiments.scripts.utils import logger

default_settings = {}


class InteractionEnergyRunner(TaskRunner):
    def __init__(
        self,
        calculator: str | None = None,
        structure: str | List[str] | None = None,
        settings: str | dict | None = None,
        **kwargs,
    ):
        super().__init__(calculator, structure, settings, default_settings, **kwargs)

    def skip(self, **kwargs):
        return False

    def task(self, atoms: ase.Atoms, settings: dict, **kwargs):

        structure_name = kwargs["structure_name"]
        output_file = f"interaction_energy.parquet"

        logger.info(f"Starting energy calculation for {structure_name}")

        df = pd.DataFrame(
            {
                "structure": [structure_name],
                "potential_energy": [atoms.get_potential_energy()],
                "forces": [atoms.get_forces().tolist()],
                "calculator": [self.calculator],
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
    args = parser.parse_args()

    runner = InteractionEnergyRunner(args.calculator, args.structure, args.settings)
    runner.run()
