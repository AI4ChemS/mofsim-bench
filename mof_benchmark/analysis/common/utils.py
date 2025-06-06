import os
import json
from functools import cache, lru_cache

import ase.atom
import pandas as pd
from pymatgen.core import Structure
from ase.build import minimize_rotation_and_translation
from pathlib import Path
from ase.io import read
from typing import Union, Any
import numpy as np
import ase
from ase.geometry import find_mic
from ase.data import covalent_radii as CR

from mof_benchmark import base_dir
from mof_benchmark.experiments.scripts.utils import logger

__local_dir = os.path.dirname(os.path.abspath(__file__))

METAL_ELEMENTS = [
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
]
METAL_NUMBERS = [ase.atom.atomic_numbers[metal] for metal in METAL_ELEMENTS]


def is_metal(number: np.ndarray[int]) -> np.ndarray[bool]:
    """
    Check if an atom is a metal

    Parameters:
    ----------
    number : np.ndarray[int]
        Atomic numbers

    Returns:
    -------
    np.ndarray[bool]
        True if the atom is a metal
    """

    return np.isin(number, METAL_NUMBERS)


def get_trajectory_rmsd(traj):
    """
    Calculate the root mean squared deviation between all frames in a trajectory

    Parameters:
    ----------
    traj : ase.Atoms
        Trajectory

    Returns:
    -------
    np.ndarray
        Root mean squared deviation between all frames
    """

    return np.array([position_rmsd(traj[0], frame) for frame in traj[0:]])


def get_reference_rmsd(traj, reference):
    """
    Calculate the root mean squared deviation between all frames in a trajectory and a reference structure

    Parameters:
    ----------
    traj : ase.Atoms
        Trajectory
    reference : ase.Atoms
        Reference structure

    Returns:
    -------
    np.ndarray
        Root mean squared deviation between all frames and the reference
    """

    return np.array([position_rmsd(reference, frame) for frame in traj[0:]])


def string2index(stridx: str) -> Union[int, slice, str]:
    """Convert index string to either int or slice"""
    if ":" not in stridx:
        # may contain database accessor
        try:
            return int(stridx)
        except ValueError:
            return stridx
    i = [None if s == "" else int(s) for s in stridx.split(":")]
    return slice(*i)


@cache
def get_qmof_df():
    qmof_path = base_dir / "experiments" / "structures" / "qmof"

    qmof_df = pd.read_parquet(
        os.path.join(qmof_path, "qmof.parquet"), columns=["qmof_id", "name"]
    )

    return qmof_df


@cache
def get_qmof_dft_structure(name: str) -> tuple[str, ase.Atoms]:
    """
    Get the DFT structure for the given structure

    Parameters:
    ----------
    name : str
        Structure file path. The file stem is used to identify the QMOF ID.

    Returns:
    -------
    str
        QMOF ID. None if no QMOF ID was found.
    ase.Atoms
        The DFT structure. None if no QMOF ID was found.
    """

    qmof_path = base_dir / "experiments" / "structures" / "qmof"

    qmof_df = get_qmof_df()

    if not any(qmof_df["name"] == name):
        logger.error(f"Could not find QMOF ID for {name}")
        return None, None

    sel = [("name", "==", name)]
    row = pd.read_parquet(
        os.path.join(qmof_path, "qmof_structure_data.parquet"), filters=sel
    )

    if len(row) == 0:
        logger.error(f"Could not find QMOF ID for {name}")
        return None, None

    qmof_id = row["qmof_id"].iloc[0]
    atoms = Structure.from_dict(json.loads(row["structure"].iloc[0])).to_ase_atoms()

    return qmof_id, atoms


@cache
def get_dft_structure(structure: str) -> tuple[str, ase.Atoms]:
    """
    Get the DFT structure for the given structure

    Parameters:
    ----------
    structure : str
        Structure file path. The file stem is used to identify the cif name.

    Returns:
    -------
    str
        cif name. None if no cif name was found.
    ase.Atoms
        The DFT structure. None if no cif name was found.
    """

    cif_ids = pd.read_csv(
        base_dir
        / "analysis"
        / "dft_data"
        / "1st_2nd_final_filter_good_bulk_modulus_calc_info_combined.csv"
    )

    if not any(cif_ids["structure"] == structure):
        logger.error(f"Could not find DFT structure for {structure}")
        return None, None

    cif_name = cif_ids[cif_ids["structure"] == structure]["cif_name"].iloc[0] + ".cif"
    cif_path = base_dir / "analysis" / "dft_data" / "set_1" / cif_name
    if not os.path.exists(cif_path):
        cif_path = base_dir / "analysis" / "dft_data" / "set_2" / cif_name
    if not os.path.exists(cif_path):
        logger.error(f"Could not find DFT structure for {structure}")
        return None, None
    atoms = read(cif_path)

    return cif_name, atoms


# @lru_cache(maxsize=100) doesnt work with lists
def get_dft_structures(structures):
    """
    Get the DFT structures for the given structures

    Parameters:
    ----------
    structures : list[str]
        List of structure file paths. The file stems are used to identify the QMOF IDs.

    Returns:
    -------
    list[str]
        List of QMOF IDs. None if no QMOF ID was found.
    list[ase.Atoms]
        List of the DFT structures. None if no QMOF ID was found.
    """

    structure_names = [Path(structure).stem for structure in structures]

    qmof_path = base_dir / "experiments" / "structures" / "qmof"

    logger.info(f"Loading QMOF data from {qmof_path}")
    with open(os.path.join(qmof_path, "qmof.json")) as f:
        qmof = json.load(f)
    qmof_df = pd.json_normalize(qmof).set_index("qmof_id")

    qmof_ids = []
    for name in structure_names:
        try:
            qmof_ids.append(qmof_df[qmof_df["name"] == name].index[0])
        except IndexError:
            logger.error(f"Could not find QMOF ID for {name}")
            qmof_ids.append(None)

    logger.info(f"Loading QMOF structure data from {qmof_path}")
    with open(os.path.join(qmof_path, "qmof_structure_data.json")) as f:
        struct_data = json.load(f)

    qmof_structs = {}
    for entry in struct_data:
        if not entry["qmof_id"] in qmof_ids:
            continue
        qmof_structs[entry["qmof_id"]] = Structure.from_dict(
            entry["structure"]
        ).to_ase_atoms()

    dft_structures = []
    for i in range(len(structure_names)):
        qmof_id = qmof_ids[i]
        if qmof_id is None:
            dft_structures.append(None)
        else:
            dft_structures.append(qmof_structs[qmof_id])

    return qmof_ids, dft_structures


def rmsd(a: np.ndarray, b: np.ndarray):
    """
    Calculate the root mean squared deviation between two arrays

    Parameters:
    ----------
    a : np.ndarray
        First array
    b : np.ndarray
        Second array

    Returns:
    -------
    float
        Root mean squared deviation
    """

    return np.linalg.norm(a - b) / np.sqrt(np.prod(a.shape))


def position_rmsd(target: ase.Atoms, atoms: ase.Atoms):
    """
    Calculate the root mean squared deviation between two structures

    Parameters:
    ----------
    target : ase.Atoms
        First structure
    atoms : ase.Atoms
        Second structure

    Returns:
    -------
    float
        Root mean squared deviation
    """

    target = target.copy()
    atoms = atoms.copy()

    minimize_rotation_and_translation(target, atoms)

    displacements, _ = find_mic(
        target.positions - atoms.positions, cell=target.cell, pbc=target.pbc
    )

    return np.sqrt(np.mean(displacements**2))


## based on https://gist.github.com/tgmaxson/8b9d8b40dc0ba4395240
def get_coordination_numbers(atoms, covalent_percent=1.25):
    """
    Returns an array of coordination numbers and an array of existing bonds
    determined by distances and covalent radii. A bond is defined as 120%
    of the combined covalent radii or less by default. This threshold can be
    adjusted using the 'covalent_percent' parameter.

    Parameters:
        atoms: ASE Atoms object
            The structure containing atomic positions and types.
        covalent_percent: float, optional
            Factor to multiply the sum of covalent radii to define a bond (default = 1.25).

    Returns:
        tuple:
            - cn (numpy.ndarray): Array of coordination numbers for each atom.
            - bonded (list of lists): Indices of bonded atoms for each atom.
    """

    # Get pairwise distances and scale by the covalent factor
    distances = atoms.get_all_distances(mic=True)

    # Atomic numbers and covalent radii
    numbers = atoms.numbers
    cr = np.take(CR, numbers)

    # Create pairwise radii sums
    radii_sums = (cr[:, None] + cr[None, :]) * covalent_percent

    # Determine bonds: bonded[i, j] is True if atoms i and j are bonded
    bonded_mask = (distances <= radii_sums) & (distances > 0)

    # Compute coordination numbers
    cn = np.sum(bonded_mask, axis=1)

    # Extract bonded indices
    bonded = [list(np.nonzero(bonded_mask[i])[0]) for i in range(len(atoms))]

    return cn, bonded


def get_coordinate_distance_by_atom_type(a: ase.Atoms, b: ase.Atoms) -> dict:

    atom_types = set(a.get_chemical_symbols())
    atom_type_distances = {}

    for atom_type in atom_types:
        atom_type_indices_a = np.where(
            np.asarray(a.get_chemical_symbols()) == atom_type
        )[0]
        atom_type_indices_b = np.where(
            np.asarray(b.get_chemical_symbols()) == atom_type
        )[0]

        atom_type_distances[atom_type] = np.sqrt(
            (
                find_mic(
                    a.positions[atom_type_indices_a] - b.positions[atom_type_indices_b],
                    cell=a.cell,
                )[0]
                ** 2
            ).mean(axis=1)
        )

    return atom_type_distances


def get_stage(traj: list[ase.Atoms]) -> ase.Atoms:
    """
    Get the structure at a specific stage in the trajectory

    Parameters:
    ----------
    traj : list[ase.Atoms]
        Trajectory

    Returns:
    -------
    stage_indices : list[int]
    """

    def get_stage_index(atoms):
        try:
            return atoms.info["stage"]
        except KeyError:
            return 0

    stage_indices = [get_stage_index(atoms) for atoms in traj]

    return stage_indices
