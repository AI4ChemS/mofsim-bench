import os
from typing import List

import ase
from ase import units
import numpy as np
from ase.eos import EquationOfState

from mof_benchmark.experiments.scripts.common.runner import TaskRunner
from mof_benchmark import base_dir
import mof_benchmark.experiments.scripts.common.optimization as OPT
from mof_benchmark.experiments.scripts.utils import logger
from mof_benchmark.experiments.scripts.utils.utils import (
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
    "bulk_modulus": {
        "criterion": {"fmax": 1e-3, "steps": 1000},
        "optimizer": "FIRE",
        "optimizer_kwargs": {},
        "eps": 0.04,
        "npoints": 5,
        "eos": "murnaghan",
    },
}


class BulkModulusRunner(TaskRunner):
    def __init__(
        self,
        calculator: str | None = None,
        structure: str | List[str] | None = None,
        settings: str | dict | None = None,
        **kwargs,
    ):
        super().__init__(calculator, structure, settings, _default_settings, **kwargs)

    def skip(self, **kwargs):

        filename = f"{kwargs['structure_name']}_bulk_modulus.npz"
        if os.path.exists(filename):
            data = np.load(filename, allow_pickle=True)
            if (
                "B" in data
                and isinstance(data["B"], np.ndarray)
                and data["B"].item() is not None
            ):
                return True

        return False

    def task(self, atoms: ase.Atoms, settings: dict, **kwargs):

        structure_name = kwargs["structure_name"]

        logger.info(f"Starting bulk modulus calculation for {structure_name}")

        stages = settings["stages"]

        volumes = None
        energies = None
        v0 = None
        e0 = None
        B = None
        eps = None
        npoints = None

        try:
            for i, stage in enumerate(stages):

                if isinstance(stage, dict):
                    stage_type = list(stage.keys())[0]
                    stage_settings = update_dicts(
                        settings[stage_type], stage[stage_type]
                    )
                else:
                    stage_type = stage
                    stage_settings = update_dicts(settings[stage_type], {})

                if stage_type == "opt":
                    # OPT stage
                    logger.info(f"Starting optimization")
                    opt_settings = stage_settings
                    OPT.run(atoms=atoms, **opt_settings)

                elif stage_type == "bulk_modulus":
                    # Bulk modulus stage
                    logger.info(f"Starting bulk modulus calculation")
                    bulk_modulus_settings = stage_settings

                    eps = bulk_modulus_settings["eps"]
                    npoints = bulk_modulus_settings["npoints"]

                    relaxed = atoms.copy()
                    relaxed.calc = atoms.calc

                    # Run bulk modulus calculation
                    volumes = []
                    energies = []

                    # scaling is applied to each cell vector
                    # V = abc
                    # V' = (a * s^(1/3)) * (b * s^(1/3)) * (c * s^(1/3)) = s * V
                    scaling_factors = np.linspace(1 - eps, 1 + eps, npoints) ** (1 / 3)

                    for scaling_factor in scaling_factors:
                        atoms = relaxed.copy()
                        atoms.calc = relaxed.calc
                        atoms.set_cell(
                            relaxed.get_cell() * scaling_factor, scale_atoms=True
                        )

                        opt_settings = stage_settings
                        OPT.run(atoms=atoms, **opt_settings)

                        volumes.append(atoms.get_volume())
                        energies.append(atoms.get_potential_energy())

                    atoms = relaxed.copy()

                    eos = EquationOfState(
                        volumes, energies, eos=bulk_modulus_settings["eos"]
                    )
                    v0, e0, B = eos.fit()

                    np.savez(
                        f"{structure_name}_bulk_modulus.npz",
                        volumes=volumes,
                        energies=energies,
                        v0=v0,
                        e0=e0,
                        eps=eps,
                        npoints=npoints,
                        B=B / units.kJ * 1.0e24,
                    )

                    logger.info(
                        f"Finished bulk modulus calculation for {structure_name}. Bulk modulus: {B / units.kJ * 1.0e24:.2f} GPa"
                    )

                    return

        except Exception as e:
            logger.error(f"Error in bulk modulus calculation for {structure_name}: {e}")

        logger.info(
            f"Bulk modulus calculation did not finish correctly for {structure_name}."
        )
        np.savez(
            f"{structure_name}_bulk_modulus.npz",
            volumes=volumes,
            energies=energies,
            v0=v0,
            e0=e0,
            eps=eps,
            npoints=npoints,
            B=B / units.kJ * 1.0e24 if B is not None else None,
        )


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

    runner = BulkModulusRunner(args.calculator, args.structure, args.settings)
    runner.run()
