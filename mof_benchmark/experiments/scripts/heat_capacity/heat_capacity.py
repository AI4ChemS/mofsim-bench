import os
from typing import Dict, List, Any

import ase
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from tqdm import tqdm

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
        # "filter": "FrechetCellFilter",
        "filter_kwargs": {},
    },
    "heat_capacity": {
        "phonopy_kwargs": {},
        "generate_displacements_kwargs": {},
        "supercell_matrix": [1, 1, 1],
        "mesh": 100,
        "t_min": 0,
        "t_max": 1000,
        "t_step": 10,
    },
}


def get_force_constants(
    structure: ase.Atoms,
    supercell_matrix: np.ndarray,
    kwargs_phonopy: Dict[str, Any] = {},
    kwargs_generate_displacements: Dict[str, Any] = {},
) -> Phonopy:
    """

    Function adapted from https://calorine.materialsmodeling.org/_modules/calorine/tools/phonons.html#get_force_constants

    Calculates the force constants for a given structure using
    `phonopy <https://phonopy.github.io/phonopy/>`_, which needs to be cited if this function
    is used for generating data for publication.
    The function returns a `Phonopy` object that can be used to calculate, e.g.,
    the phonon dispersion, the phonon density of states as well as related quantities such
    as the thermal displacements and the free energy.

    Parameters
    ----------
    structure
        structure for which to compute the phonon dispersion; usually this is a primitive cell. Needs to have
        a calculator attached.
    supercell_matrix
        specification of supercell size handed over to phonopy;
        should be a tuple of three values or a matrix
    kwargs_phonopy
        *Expert option*:
        keyword arguments used when initializing the `Phonopy` object;
        this includes, e.g., the tolerance used when determining the symmetry (`symprec`) and
        `parameters for the non-analytical corrections
        <https://phonopy.github.io/phonopy/phonopy-module.html#non-analytical-term-correction>`_
        (`nac_params`)
    kwargs_generate_displacements
        *Expert option*:
        keyword arguments to be handed over to the `generate_displacements` method;
        this includes in particular the `distance` keyword, which specifies the
        magnitude of the atomic displacement imposed when calculating the force constant matrix
    """

    # prepare primitive unit cell for phonopy
    structure_ph = PhonopyAtoms(
        symbols=structure.symbols,
        cell=structure.cell,
        scaled_positions=structure.get_scaled_positions(),
    )

    # make sure we are using the masses intended by the user
    structure_ph.masses = structure.get_masses()

    # prepare supercells
    phonon = Phonopy(structure_ph, supercell_matrix, **kwargs_phonopy)
    phonon.generate_displacements(**kwargs_generate_displacements)

    # compute force constant matrix
    logger.info("Calculating forces for force constant matrix")
    forces = []
    for structure_ph in tqdm(phonon.supercells_with_displacements):
        structure_ase = ase.Atoms(
            symbols=structure_ph.symbols,
            cell=structure_ph.cell,
            scaled_positions=structure_ph.scaled_positions,
            pbc=structure.pbc,
        )
        structure_ase.calc = structure.calc
        forces.append(structure_ase.get_forces().copy())

    phonon.forces = forces
    phonon.produce_force_constants()

    return phonon


class HeatCapacityRunner(TaskRunner):
    def __init__(
        self,
        calculator: str | None = None,
        structure: str | List[str] | None = None,
        settings: str | dict | None = None,
        **kwargs,
    ):
        super().__init__(calculator, structure, settings, _default_settings, **kwargs)

    def skip(self, **kwargs):
        return os.path.exists(f"{kwargs['structure_name']}_thermal_properties.npz")

    def task(self, atoms: ase.Atoms, settings: dict, **kwargs):

        structure_name = kwargs["structure_name"]

        logger.info(f"Starting heat capacity calculation for {structure_name}")

        stages = settings["stages"]

        for i, stage in enumerate(stages):

            if isinstance(stage, dict):
                stage_type = list(stage.keys())[0]
                stage_settings = update_dicts(settings[stage_type], stage[stage_type])
            else:
                stage_type = stage
                stage_settings = update_dicts(settings[stage_type], {})

            if stage_type == "opt":
                # OPT stage
                logger.info(f"Starting optimization")
                opt_settings = stage_settings
                OPT.run(atoms=atoms, **opt_settings)

            elif stage_type == "heat_capacity":
                # Heat capacity stage
                logger.info(f"Starting heat capacity calculation")
                heat_capacity_settings = stage_settings

                # Get force constants
                phonon = get_force_constants(
                    atoms,
                    supercell_matrix=heat_capacity_settings["supercell_matrix"],
                    kwargs_phonopy=heat_capacity_settings["phonopy_kwargs"],
                    kwargs_generate_displacements=heat_capacity_settings[
                        "generate_displacements_kwargs"
                    ],
                )

                # Calculate heat capacity
                logger.info(f"Running phonon mesh calculations.")
                phonon.run_mesh(mesh=heat_capacity_settings["mesh"])
                logger.info(f"Running thermal properties calculations.")
                phonon.run_thermal_properties(
                    t_step=heat_capacity_settings["t_step"],
                    t_max=heat_capacity_settings["t_max"],
                    t_min=heat_capacity_settings["t_min"],
                )

                # Save results
                phonon.save(
                    f"{structure_name}_phonopy.yaml", settings={"force_constants": True}
                )
                tp_dict = phonon.get_thermal_properties_dict()
                np.savez(
                    f"{structure_name}_thermal_properties.npz",
                    temperatures=tp_dict["temperatures"],
                    heat_capacity_mol=tp_dict["heat_capacity"],
                    heat_capacity_g=(
                        tp_dict["heat_capacity"] / phonon.primitive.masses.sum()
                    ),
                    entropy=tp_dict["entropy"],
                    free_energy=tp_dict["free_energy"],
                    t_min=heat_capacity_settings["t_min"],
                    t_max=heat_capacity_settings["t_max"],
                    t_step=heat_capacity_settings["t_step"],
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

    runner = HeatCapacityRunner(args.calculator, args.structure, args.settings)
    runner.run()
