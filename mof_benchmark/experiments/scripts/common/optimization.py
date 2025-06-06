import time
import ase.calculators
from ase.optimize import (
    BFGS,
    LBFGS,
    MDMin,
    FIRE,
    LBFGSLineSearch,
    BFGSLineSearch,
    QuasiNewton,
    GPMin,
    CellAwareBFGS,
    ODE12r,
)
from ase.filters import (
    Filter,
    UnitCellFilter,
    ExpCellFilter,
    StrainFilter,
    FrechetCellFilter,
)
import ase
import ase.io
from ase.optimize.optimize import Optimizer

import os

from mof_benchmark.experiments.scripts.utils import logger


_valid_filters: dict[str, Filter] = {
    "Filter": Filter,
    "UnitCellFilter": UnitCellFilter,
    "ExpCellFilter": ExpCellFilter,
    "StrainFilter": StrainFilter,
    "FrechetCellFilter": FrechetCellFilter,
}

_valid_optimizers: dict[str, Optimizer] = {
    "MDMin": MDMin,
    "FIRE": FIRE,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "BFGS": BFGS,
    "BFGSLineSearch": BFGSLineSearch,
    "QuasiNewton": QuasiNewton,
    "GPMin": GPMin,
    "CellAwareBFGS": CellAwareBFGS,
    "ODE12r": ODE12r,
}


def get_optimizer(optimizer: str, **optimizer_kwargs) -> Optimizer:
    """
    Get the optimizer.

    Parameters
    ----------
    optimizer : str
        Optimizer to use.

    Returns
    -------
    Optimizer
        Optimizer object.
    """
    if optimizer not in _valid_optimizers:
        raise ValueError(
            f"Invalid optimizer: {optimizer}, valid optimizers are {_valid_optimizers.keys()}"
        )
    return _valid_optimizers[optimizer](**optimizer_kwargs)


def run(
    atoms: ase.Atoms,
    optimizer: str = "BFGS",
    optimizer_kwargs: dict = {},
    filter: str = None,
    filter_kwargs: dict = {},
    criterion: dict = None,
    stage: int = 0,
    **kwargs,
) -> ase.Atoms:
    """
    Relax the structure using the given calculator.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object representing the structure to relax. Needs to have a calculator attached.
    optimizer : str
        Optimizer to use for relaxation.

    Returns
    -------
    ase.Atoms
        Relaxed structure.
    """

    # Load optimizer and filter
    if isinstance(filter, str):
        if filter not in _valid_filters:
            raise ValueError(
                f"Invalid filter: {filter}, valid filters are {_valid_filters.keys()}"
            )
        filter = _valid_filters[filter]

    if isinstance(optimizer, str):
        if optimizer not in _valid_optimizers:
            raise ValueError(
                f"Invalid optimizer: {optimizer}, valid optimizers are {_valid_optimizers.keys()}"
            )
        optimizer = _valid_optimizers[optimizer]

    assert atoms.calc is not None, "Atoms object must have a calculator attached"
    assert issubclass(optimizer, Optimizer), f"Invalid optimizer {optimizer}"
    assert filter is None or issubclass(filter, Filter), f"Invalid filter {filter}"

    filter_kwargs = filter_kwargs or {}
    optimizer_kwargs = optimizer_kwargs or {}
    if "append_trajectory" not in optimizer_kwargs:
        optimizer_kwargs["append_trajectory"] = kwargs.get("append_trajectory", False)

    filtered_atoms = filter(atoms, **filter_kwargs) if filter is not None else atoms
    if filter is not None:
        logger.info(f"Applying filter {filter} with kwargs {filter_kwargs}")
    opt = optimizer(filtered_atoms, **optimizer_kwargs)  #
    logger.info(f"Starting optimization with {optimizer} and kwargs {optimizer_kwargs}")

    def add_metadata(dyn=opt):
        dyn.atoms.info["stage"] = stage

    def check_exploded(dyn=opt):
        if abs(dyn.atoms.calc.get_potential_energy()) > 100000:
            logger.warning("System exploded. Aborting optimization.")
            raise ValueError("System exploded")

    opt.step_start_time = time.time()

    def execution_too_slow(dyn=opt):
        if (
            time.time() - dyn.step_start_time > 50
        ):  # 5 seconds per optimization step should be more than enough
            logger.warning("Optimization step took too long. Aborting optimization.")
            raise ValueError("Optimization step took too long")

        dyn.step_start_time = time.time()

    opt.attach(add_metadata, interval=1)
    opt.attach(check_exploded, interval=10)
    opt.attach(execution_too_slow, interval=10)

    opt.run(**criterion)

    return atoms
