import os
from typing import List
from functools import partial

import ase
from ase import units
from ase.md import MDLogger

from mof_benchmark.experiments.scripts.common.runner import TaskRunner
from mof_benchmark import base_dir
import mof_benchmark.experiments.scripts.common.optimization as OPT
import mof_benchmark.experiments.scripts.common.md as MD
from mof_benchmark.experiments.scripts.utils import logger
from mof_benchmark.experiments.scripts.utils.utils import (
    convert_trajectory,
    update_dicts,
)

_default_settings = {
    "output_dir": base_dir / "experiments" / "outputs",
    "opt": {
        "criterion": {"fmax": 1e-3, "steps": 1000},
        "optimizer": "BFGS",
        "optimizer_kwargs": {},
        "filter": None,
        "filter_kwargs": {},
    },
    "nvt": {
        "dynamics": "langevin",
        "total_steps": 1000,
        "initial_temperature": 300,
        "append_trajectory": True,
        "trajectory_interval": 100,
        "stage": 1,
        "ase_md_kwargs": {
            "timestep": 1.0,
            "friction": 0.002,
            "temperature_K": 300,
        },
    },
    "npt": {
        "dynamics": "npt",
        "total_steps": 20_000,
        "append_trajectory": True,
        "trajectory_interval": 100,
        "stage": 2,
        "ase_md_kwargs": {
            "timestep": 1.0,
            "externalstress": 1.0,
            "temperature_K": 300,
            "ttime": 100,
            "ptime": 1000,
            "B": 20,
        },
    },
    "nptberendsen": {
        "dynamics": "nptberendsen",
        "total_steps": 20_000,
        "append_trajectory": True,
        "trajectory_interval": 100,
        "stage": 2,
        "ase_md_kwargs": {
            "timestep": 1.0,
            "pressure_au": 1.0,
            "temperature_K": 300,
            "taut": 100,
            "taup": 1000,
            "B": 20,
        },
    },
    "isotropicmtknpt": {
        "dynamics": "isotropicmtknpt",
        "total_steps": 20_000,
        "append_trajectory": True,
        "trajectory_interval": 100,
        "stage": 2,
        "ase_md_kwargs": {
            "timestep": 1.0,
            "pressure_au": 1.0,
            "temperature_K": 300,
            "tdamp": 100,
            "pdamp": 1000,
        },
    },
}


class StabilityRunner(TaskRunner):
    def __init__(
        self,
        calculator: str | None = None,
        structure: str | List[str] | None = None,
        settings: str | dict | None = None,
        index: int | None = None,
        **kwargs,
    ):
        super().__init__(
            calculator, structure, settings, _default_settings, index, **kwargs
        )

    def skip(self, **kwargs):
        return False  # if os.path.exists(kwargs.get("structure_name") + ".traj")

    def task(self, atoms: ase.Atoms, settings: dict, **kwargs):

        structure_name = kwargs["structure_name"]
        trajectory_file = f"{structure_name}.traj"

        logger.info(f"Starting stability calculation for {structure_name}")

        stages = settings["stages"]

        for i, stage in enumerate(stages):

            if isinstance(stage, dict):
                stage_type = list(stage.keys())[0]
                if stage_type in settings:
                    stage_settings = update_dicts(
                        settings[stage_type], stage[stage_type]
                    )
                else:
                    stage_settings = stage[stage_type]
            else:
                stage_type = stage
                if stage_type in settings:
                    stage_settings = settings[stage_type]
                else:
                    logger.warning(
                        f"No settings found for stage {stage_type}. Skipping."
                    )
                    continue

            if stage_type == "opt":
                # OPT stage
                logger.info(f"Starting optimization")
                opt_settings = stage_settings
                opt_settings["optimizer_kwargs"] = opt_settings.get(
                    "optimizer_kwargs", {}
                )
                opt_settings["optimizer_kwargs"]["trajectory"] = trajectory_file
                OPT.run(atoms=atoms, **opt_settings)

            elif stage_type == "nvt":
                # NVT stage
                logger.info(f"Starting NVT simulation")
                nvt_settings = stage_settings
                nvt_settings["trajectory"] = trajectory_file
                ase_md_kwargs = nvt_settings["ase_md_kwargs"]
                ase_md_kwargs["timestep"] = ase_md_kwargs["timestep"] * units.fs
                ase_md_kwargs["friction"] = ase_md_kwargs["friction"] / units.fs
                nvt_settings["attachments"] = nvt_settings.get("attachments", []) + [
                    (partial(MDLogger, logfile="-"), 100)
                ]
                MD.run(atoms=atoms, **nvt_settings)

            elif stage_type == "npt":
                # NPT stage
                logger.info(f"Starting NPT simulation")
                npt_settings = stage_settings
                npt_settings["trajectory"] = trajectory_file
                ase_md_kwargs = npt_settings["ase_md_kwargs"]
                ase_md_kwargs["timestep"] = ase_md_kwargs["timestep"] * units.fs
                ase_md_kwargs["externalstress"] = (
                    ase_md_kwargs["externalstress"] * units.bar
                )
                ase_md_kwargs["ttime"] = ase_md_kwargs["ttime"] * units.fs
                assert (
                    "ptime" in ase_md_kwargs
                    and "B" in ase_md_kwargs
                    or "pfactor" in ase_md_kwargs
                ), "Either ptime and B or pfactor must be provided"
                if "ptime" in ase_md_kwargs:
                    ptime = ase_md_kwargs["ptime"] * units.fs
                    ase_md_kwargs.pop("ptime")
                if "B" in ase_md_kwargs:
                    B = ase_md_kwargs["B"] * units.GPa
                    ase_md_kwargs.pop("B")
                ase_md_kwargs["pfactor"] = ase_md_kwargs.get("pfactor", ptime**2 * B)
                npt_settings["attachments"] = npt_settings.get("attachments", []) + [
                    (partial(MDLogger, logfile="-"), 500)
                ]
                MD.run(atoms=atoms, **npt_settings)

            elif stage_type == "nptberendsen":
                npt_settings = stage_settings
                npt_settings["trajectory"] = trajectory_file
                ase_md_kwargs = npt_settings["ase_md_kwargs"]
                ase_md_kwargs["timestep"] = ase_md_kwargs["timestep"] * units.fs
                ase_md_kwargs["pressure_au"] = ase_md_kwargs["pressure_au"] * units.bar
                ase_md_kwargs["taut"] = ase_md_kwargs["taut"] * units.fs
                ase_md_kwargs["taup"] = ase_md_kwargs["taup"] * units.fs
                if "B" in ase_md_kwargs:
                    B = ase_md_kwargs["B"] * units.GPa
                    ase_md_kwargs.pop("B")
                    ase_md_kwargs["compressibility_au"] = 1 / B
                npt_settings["attachments"] = npt_settings.get("attachments", []) + [
                    (partial(MDLogger, logfile="-"), 500)
                ]
                MD.run(atoms=atoms, **npt_settings)

            elif stage_type == "isotropicmtknpt":
                npt_settings = stage_settings
                npt_settings["trajectory"] = trajectory_file
                ase_md_kwargs = npt_settings["ase_md_kwargs"]
                ase_md_kwargs["timestep"] = ase_md_kwargs["timestep"] * units.fs
                ase_md_kwargs["pressure_au"] = ase_md_kwargs["pressure_au"] * units.bar
                ase_md_kwargs["tdamp"] = ase_md_kwargs["tdamp"] * units.fs
                ase_md_kwargs["pdamp"] = ase_md_kwargs["pdamp"] * units.fs
                npt_settings["attachments"] = npt_settings.get("attachments", []) + [
                    (partial(MDLogger, logfile="-"), 500)
                ]
                MD.run(atoms=atoms, **npt_settings)


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

    runner = StabilityRunner(args.calculator, args.structure, args.settings, args.index)
    runner.run()
