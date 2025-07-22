from pathlib import Path
import shutil
import signal
from typing import Union
import os
import uuid
import collections.abc

from cycler import V
from scipy.linalg import schur
import ase

from numpy import var
from mof_benchmark.experiments.scripts.utils import logger
from mof_benchmark import base_dir


def update_dicts(*args):
    """
    Update multiple dictionaries. The dictionaries are updated in the order they are passed, i.e., the last dictionary
    has the highest priority.

    Parameters
    ----------
    *args
        Dictionaries to update.

    Returns
    -------
    dict
        Updated dictionary.
    """

    final_dict = {}
    for d in args:
        final_dict = _update_two_dicts(final_dict, d)
    return final_dict


def _update_two_dicts(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _update_two_dicts(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def make_upper_triangular_cell(atoms: ase.Atoms):
    u, _ = schur(atoms.get_cell(complete=True), output="complex")
    atoms.set_cell(u.real, scale_atoms=True)


def get_path(path: str, variables: dict = {}, **kwargs) -> str:
    """
    Get the path with environment and custom variables expanded.
    Always expands {base_dir} to point to the base directory of the package.

    Parameters
    ----------
    path : str
        Path with variables.
    variables : dict
        Variables to expand.

    Returns
    -------
    str
        Path with the variables expanded.
    """

    variables = variables or {}
    variables.update(kwargs)

    return os.path.expandvars(path.format(**os.environ, **variables, base_dir=base_dir))


def structure_to_files(structure: Union[str, list[str]]) -> list[str]:
    """
    Write the structure to a file.

    Parameters
    ----------
    structure : Union[str, list[str]]
        Path to the structure file or list of paths to structure files.

    Returns
    -------
    list[str]
        List of paths to the structure files.
    """
    if isinstance(structure, Path):
        return [str(structure)]
    elif isinstance(structure, str):
        if (
            os.path.isabs(structure)
            and os.path.exists(structure)
            and os.path.isfile(structure)
        ):
            return [structure]
        elif os.path.exists(structure) and os.path.isfile(structure):
            return [os.path.abspath(structure)]
        elif os.path.exists(
            base_dir / "experiments" / "structures" / structure
        ) and os.path.isfile(base_dir / "experiments" / "structures" / structure):
            return [base_dir / "experiments" / "structures" / structure]
        else:
            import yaml

            yaml_path = base_dir / "experiments" / "structures" / "structures.yaml"
            structures = yaml.safe_load(open(yaml_path))
            structures_dir = structures.get(
                "structures_dir", f"{base_dir}/experiments/structures"
            )
            if structure in structures.keys():
                return structure_to_files(
                    structures[structure]
                    # list(
                    #     map(
                    #         lambda rel_path: get_path(
                    #             os.path.join(structures_dir, rel_path)
                    #         ),
                    #         structures[structure],
                    #     )
                    # )
                )

            raise ValueError(f"File or structure shortcut {structure} not found.")

    else:
        return [path for s in structure for path in structure_to_files(s)]


class TmpWorkdir:
    def __init__(
        self, output_dir: str, tmp_dir: str | None = None, copy_outputs: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.tmp_dir = (
            Path(tmp_dir) / str(uuid.uuid4().hex) if tmp_dir is not None else None
        )
        self.copy_outputs = copy_outputs

    def __str__(self):
        return os.getcwd()

    def __enter__(self):
        self.old_cwd = os.getcwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.tmp_dir is not None:
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(self.tmp_dir)
            signal.signal(signal.SIGTERM, self.__interrupt)
            signal.signal(signal.SIGCONT, self.__interrupt)

            if self.copy_outputs and os.path.exists(self.output_dir):
                self.__copy_dir(self.output_dir, self.tmp_dir)
        else:
            os.chdir(self.output_dir)

    def __interrupt(self, signum, frame):
        if self.tmp_dir is not None:
            logger.error(
                f"Interrupted, copying files from {self.tmp_dir} to {self.output_dir}"
            )
            self.__copy_dir(self.tmp_dir, self.output_dir)
            shutil.rmtree(self.tmp_dir)
        raise KeyboardInterrupt

    def __copy_dir(self, src: Path, dst: Path):
        for el in src.iterdir():
            if el.is_file():
                shutil.copy2(el, dst)
            elif el.is_dir():
                os.makedirs(dst / el.name, exist_ok=True)
                self.__copy_dir(el, dst / el.name)
        # shutil.copytree(src, dst, dirs_exist_ok=True)

    def __exit__(self, *args):
        os.chdir(self.old_cwd)
        if self.tmp_dir is not None:
            logger.info(
                f"Exiting tempdir. Copying files from {self.tmp_dir} to {self.output_dir}"
            )
            self.__copy_dir(self.tmp_dir, self.output_dir)
            shutil.rmtree(self.tmp_dir)


def convert_trajectory(
    trajectory: str | ase.Atoms | list[ase.Atoms],
    output_file: str,
    delete: bool = False,
):
    """
    Convert the trajectory to a file.

    Parameters
    ----------
    trajectory : str | ase.Atoms | list[ase.Atoms]
        Trajectory to convert.
    output_file : str
        Output file.
    """

    format = Path(output_file).suffix[1:]

    if isinstance(trajectory, str):
        traj = ase.io.read(trajectory, ":")
    elif isinstance(trajectory, ase.Atoms):
        traj = [trajectory]
    else:
        traj = trajectory

    ase.io.write(output_file, traj, format=format)

    if delete and isinstance(trajectory, str):
        os.remove(trajectory)
