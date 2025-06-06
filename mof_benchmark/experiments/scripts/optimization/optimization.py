from typing import List
import os

import ase
import ase.io

from mof_benchmark.experiments.scripts.common.runner import TaskRunner
from mof_benchmark import base_dir
import mof_benchmark.experiments.scripts.common.optimization as OPT


_default_settings = {
    "criterion": {"fmax": 1e-3, "steps": 1000},
    "output_dir": base_dir / "experiments" / "outputs",
    "optimizer": "BFGS",
    "optimizer_kwargs": {},
    "filter": "FrechetCellFilter",
    "filter_kwargs": {},
}


class OptimizationRunner(TaskRunner):
    def __init__(
        self,
        calculator: str | None = None,
        structure: str | List[str] | None = None,
        settings: str | dict | None = None,
        **kwargs,
    ):
        super().__init__(calculator, structure, settings, _default_settings, **kwargs)

    def skip(self, **kwargs):
        return os.path.exists(kwargs.get("structure_name") + ".cif")

    def task(self, atoms: ase.Atoms, settings: dict, **kwargs):

        structure_name = kwargs["structure_name"]

        settings["optimizer_kwargs"] = settings.get("optimizer_kwargs") or {}
        settings["optimizer_kwargs"]["trajectory"] = f"{structure_name}.traj"

        OPT.run(atoms=atoms, **settings)

        ase.io.write(structure_name + ".cif", atoms)


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

    runner = OptimizationRunner(args.calculator, args.structure, args.settings)
    runner.run()
