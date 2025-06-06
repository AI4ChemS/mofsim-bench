from ase import units
import ase
from ase.md import VelocityVerlet, Langevin, Andersen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.npt import NPT
from ase.md.md import MolecularDynamics
from ase.io.trajectory import TrajectoryWriter
from ase.io import read
import ase.optimize.optimize
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
import numpy as np


from mof_benchmark.experiments.scripts.utils.utils import get_path
from mof_benchmark.experiments.scripts.utils import logger

import os
import argparse

_valid_dynamics: dict = {
    "velocityverlet": VelocityVerlet,
    "andersen": Andersen,
    "nvtberendsen": NVTBerendsen,
    "langevin": Langevin,
    "npt": NPT,
    "nptberendsen": NPTBerendsen,
    "isotropicmtknpt": IsotropicMTKNPT,
}


# Trajectory writer helper class
class PropertyWriter:
    def __init__(self, dyn, property_csv, properties=None, dt=1 * units.fs):
        self.dyn = dyn
        self.property_csv = property_csv
        self.properties = properties
        self.dt = dt

    def __call__(self):

        if not os.path.exists(self.property_csv):
            with open(self.property_csv, "w") as f:
                f.write("step," + ",".join(self.properties) + "\n")

        with open(self.property_csv, "a") as f:
            step = int(round(self.dyn.get_time() / self.dt))

            line = f"{step}"

            for prop in self.properties:

                if prop == "temperature":
                    value = self.dyn.atoms.get_temperature()
                elif prop == "potential_energy":
                    value = self.dyn.atoms.get_potential_energy()
                elif prop == "kinetic_energy":
                    value = self.dyn.atoms.get_kinetic_energy()
                elif prop == "total_energy":
                    value = self.dyn.atoms.get_total_energy()
                elif prop == "pressure":
                    value = self.dyn.atoms.get_pressure()
                elif prop == "volume":
                    value = self.dyn.atoms.get_volume()
                else:
                    raise ValueError(f"Invalid property {prop}")

                line += f",{value}"

            line += "\n"
            f.write(line)


def run(
    atoms: ase.Atoms,
    dynamics: str,
    total_steps: int,
    ase_md_kwargs: dict,
    initial_temperature: float | None = None,
    md_velocity_seed: int = 0,
    zero_linear_momentum: bool = True,
    zero_angular_momentum: bool = True,
    trajectory: str | None = None,
    trajectory_interval: int = 1,
    append_trajectory: bool = False,
    trajectory_properties: list[str] | None = None,
    attachments: list[tuple[type, int]] = [],
    restart: bool = False,
    stage: int = 0,
):
    """
    Run a single stage of the MD simulation.

    Parameters
    ----------
    step : int
        Current step of the simulation.
    atoms : ase.Atoms
        Atoms object to run the simulation on.
    simulation_stage : dict
        Dictionary containing the simulation type and its settings.
    global_settings : dict
        Global settings for the simulation.
    """

    step = 0
    restart_step = 0

    if trajectory is not None:

        mode = "w" if not append_trajectory else "a"

        if os.path.exists(trajectory):
            if restart:
                last_atoms = read(trajectory, "-1")
                last_stage = last_atoms.info["stage"]
                if last_stage > stage:
                    logger.warning(
                        f"Trajectory file {trajectory} already contains a stage {last_stage} greater than the current stage {stage}. Continuing to next stage."
                    )
                    return
                step = last_atoms.info["step"]
                restart_step = step
                atoms = last_atoms
                mode = "a"
            elif not append_trajectory:
                logger.warning(
                    f"Trajectory file {trajectory} already exists. Overwriting."
                )

        kwargs = {}
        # if trajectory_properties is not None:
        kwargs["properties"] = trajectory_properties

        logger.info(f"Writing trajectory to {trajectory}.")
        if append_trajectory:
            logger.info("Appending to trajectory.")
        traj = TrajectoryWriter(
            trajectory,
            mode,
            atoms,
            **kwargs,
        )

    if dynamics not in _valid_dynamics:
        raise ValueError(
            f"Invalid dynamics {dynamics}. Available dynamics are {_valid_dynamics.keys()}."
        )
    dynamics = _valid_dynamics[dynamics]

    if step == 0 and initial_temperature is not None:
        logger.info(
            f"Setting initial velocities with seed {md_velocity_seed} and temperature {initial_temperature} K."
        )
        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=initial_temperature,
            rng=np.random.default_rng(seed=md_velocity_seed),
        )

        if zero_linear_momentum:
            logger.info("Setting the center-of-mass momentum to zero.")
            Stationary(atoms)
        if zero_angular_momentum:
            logger.info("Setting the total angular momentum to zero.")
            ZeroRotation(atoms)

    dyn = dynamics(atoms, **ase_md_kwargs)

    def add_metadata(dyn: MolecularDynamics = dyn):
        dyn.atoms.info["stage"] = stage
        dyn.atoms.info["step"] = step
        dyn.atoms.info["restart_step"] = restart_step

    dyn.attach(add_metadata, interval=1)

    for attachment, interval in attachments:
        logger.info(f"Attaching {attachment} with interval {interval}.")
        dyn.attach(attachment(dyn=dyn, atoms=atoms), interval)
    if trajectory is not None:
        dyn.attach(traj.write, interval=trajectory_interval)

    logger.info(f"Starting MD simulation with {dynamics} and kwargs {ase_md_kwargs}")

    while step < total_steps:
        dyn.run(1)
        step += 1

    return step


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="MD",
        description="MD tools for versatile simulations",
    )

    parser.add_argument("settings")
    parser.add_argument("-c", "--calculator", default=None)

    args = parser.parse_args()
    filtered_args = {k: v for k, v in vars(args).items() if v is not None}

    run(args.settings, **filtered_args)
