from ase.io import read
import sys
from mof_benchmark.setup.calculator import get_calculator
from mof_benchmark.experiments.scripts.speed import benchmark
from ase.optimize import LBFGS
from ase.md.nose_hoover_chain import IsotropicMTKNPT
import time


def main(calc_name):
    calc_name = sys.argv[1]
    calc = get_calculator(calc_name)
    atoms = read("../experiments/structures/main_set/MOF-5.cif")
    atoms.calc = calc
    print("Energy:", atoms.get_potential_energy())
    print("Forces:", atoms.get_forces())
    if "stress" in calc.implemented_properties:
        print("Stress:", atoms.get_stress())
    else:
        print("Stress not implemented.")
        print("Some properties of the benchmark require stress calculations.")
        print("Please ensure that the calculator supports the stress property.")
        return

    a = atoms.copy()
    a.calc = calc
    dyn = LBFGS(a)
    steps = 100
    start_time = time.time()
    dyn.run(fmax=1e-10, steps=steps)
    end_time = time.time()
    print(
        f"Optimization time per step: {(end_time - start_time) / dyn.nsteps * 1000:.2f} ms"
    )

    num, duration = benchmark(calc, atoms)
    print(f"Duration per step: {duration / num * 1000:.2f} ms ({num} it)")
    print(f"Test completed successfully: {calc_name}.")

    return


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python calculator_test.py <calc_name>")
        print("  Refer to the README for calculator setup instructions")
        sys.exit(1)

    main(sys.argv[1])
