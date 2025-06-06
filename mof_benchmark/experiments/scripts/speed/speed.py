from ase.calculators.mixing import LinearCombinationCalculator
from ase.calculators.calculator import all_changes
import timeit
import numpy as np
import time
from ase.optimize import FIRE
from ase.io import read
from ase.filters import FrechetCellFilter
from mof_benchmark.setup.calculator import get_calculator
from mof_benchmark import base_dir


def get_values(calc, atoms, properties):
    if isinstance(calc, LinearCombinationCalculator):
        for calc in calc.mixer.calcs:
            calc.results = {}
    else:
        calc.results = {}
    atoms = atoms.copy()
    atoms.positions += np.random.normal(scale=1e-4, size=atoms.positions.shape)
    res = calc.calculate(atoms, properties, all_changes)
    return res


def benchmark(calc, atoms, properties=["energy", "forces", "stress"]):
    # duration = timeit.timeit(lambda: get_values(calc, atoms, properties), number=n)
    # return duration / n

    i = 1
    while True:
        for j in 1, 2, 5:
            number = i * j
            time_taken = timeit.timeit(
                lambda: get_values(calc, atoms, properties), number=number
            )
            if time_taken >= 10:
                return (number, time_taken)
        i *= 10


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--calculator", type=str, required=True)
    args = parser.parse_args()

    calc = get_calculator(args.calculator)

    atoms = read("../../structures/main_set/MOF-5.cif")
    a = atoms.copy()
    a.calc = calc
    a = FrechetCellFilter(a)
    dyn = FIRE(a)
    steps = 1000
    # warmup in case the calc needs it
    dyn.run(fmax=0, steps=1)
    start_time = time.time()
    dyn.run(fmax=0, steps=steps)
    end_time = time.time()
    print(
        f"Optimization time per step: {(end_time - start_time) / (dyn.nsteps - 1) * 1000:.2f} ms"
    )
    output_dir = base_dir / "outputs" / "speed"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{args.calculator}.npz",
        total_time=end_time - start_time,
        steps=steps,
        time_per_step=(end_time - start_time) / (dyn.nsteps - 1),
    )
