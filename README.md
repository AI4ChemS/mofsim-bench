# MOFSimBench: Benchmarking Universal Machine Learning Interatomic Potentials for Metal-Organic Frameworks

[![arXiv](https://img.shields.io/badge/arXiv-2507.11806-b31b1b.svg)](https://arxiv.org/abs/2507.11806)

This repository contains the code and data for the paper "MOFSimBench: Evaluating Universal Machine Learning Interatomic Potentials for Metal-Organic Framework Molecular Modeling". The project aims to benchmark the performance of various Universal Machine Learning Interatomic Potentials (uMLIPs) in simulating Metal-Organic Frameworks (MOFs) across different properties, including structural optimization, simulation stability, and bulk modulus and heat capacity.


## Table of Contents

- [:hammer_and_wrench: Installation](#hammer_and_wrench-installation)
- [:gear: Setting up your calculator](#gear-setting-up-your-calculator)
- [:file_folder: Project Structure](#file_folder-project-structure)
- [:rocket: Running the benchmark](#rocket-running-the-benchmark)
- [:bar_chart: Analyzing the results](#bar_chart-analyzing-the-results)
- [:handshake: Contributing](#handshake-contributing)


## :hammer_and_wrench: Installation

The recommended way to run the benchmark is via Conda environments and SLURM. The SLURM scripts expect environments to be named `mb_your-model`.

1. **Create a Conda environment:**

```bash
conda create -n mb_your-model python=3.11 # or other Python version
conda activate mb_your-model
```

2. **Clone the repository:**

```bash
git clone https://github.com/AI4ChemS/mof-umlip-benchmark
cd mof-umlip-benchmark
```

3. **Install core dependencies:**

```bash
pip install .
```

4. **Install DFTD3 package (if needed for D3 corrections):**

```bash
pip install torch-dftd
```

## :gear: Setting up your calculator

Set up your calculator in the `mof_benchmark/setup/calculator.yaml` and `mof_benchmark/setup/calculator.py` files.

Models normally require inference-time D3 corrections; make sure to enable them for your model in the `yaml` file. A typical entry could look like this:

```yml
orb_v3:
  model_name: orb-v3-conservative-inf-omat
  with_d3: true
  model_kwargs:
    precision: float32-highest
```

Note: The calculator name is used to identify the Conda environment. For an environment named `mb_your-model`, its name is expected to be `your-model[_suffix]` with an optional suffix that is not used in the identification of the Conda environment.

For the example above, the model name is `orb` and the suffix is `_v3`. The scripts expect the corresponding Conda environment to be `mb_orb`.

Connect your model to the benchmark in the `calculator.py` file. Several architectures from the study are implemented already.

To test that the model works, run:
```bash
python mof_benchmark/setup/test_calculator.py your-model
```

It should output energy, forces, and stresses, run a short optimization, and a quick speed test.

## :file_folder: Project Structure


The repository is organized as follows:


- `mof_benchmark/`: Contains the core Python package.
	- `analysis/`: Scripts and Streamlit pages for analyzing and visualizing results.
	- `experiments/`: Scripts and configurations for running tasks.
		- `scripts/`: Python scripts for different experiments (optimization, stability, heat capacity, bulk modulus).
		- `structures/`: MOF structure definitions.
	- `setup/`: Configuration files for calculators (e.g., `calculators.yaml`, `calculator.py`).


## :rocket: Running the benchmark

The benchmark is optimized to run on distributed systems managed with SLURM and can be run with a single command. On different systems, each task can also be easily called using the respective Python scripts.


### Slurm

Sample files are available for each task. They can be found under `mof_benchmark/experiments/scripts` in the respective task folders. The slurm submission scripts are named `submit.sh`. Adapt them to the required settings on your HPC.

For stability MDs, each structure is submitted in a separate job due to the extended runtime. In this case, the submit script relies on SLURM arrays to distribute the jobs.

With correctly configured `submit.sh` files in the `bulk_modulus`, `heat_capacity`, `optimization`, and `stability` directories, all jobs can easily be submitted via the `run_all.sh` script in `mof_benchmark/experiments/scripts`:

```bash
./run_all.sh your-calculator
```


### Python

All tasks can also be run from Python directly (The SLURM scripts just call these):

E.g., for the optimization task, run:
```bash
python optimization.py --calculator your-model --settings optimization.yaml

```

For the stability tasks, run:
```bash
python stability.py --calculator your-model --settings stability_prod_mtk.yaml --index 0
```

The index (0-99) specifies the structure to run.


### QMOF energy comparison task

As an additional test, we compare the energy predictions of uMLIPs to QMOF DFT references. To run this task, you have to download the QMOF database and place the `qmof_database` folder in `mof_benchmark/experiments/structures`.

### Interaction energy task

To perform the task and analysis, the GoldDAC `test.xyz` file must be placed in the `mof_benchmark/analysis/interaction_energy` directory.
Structures must also be extracted into the `mof_benchmark/experiments/structures/golddac` directory. A python notebook is provided to extract the structures from the `test.xyz` file.


## :bar_chart: Analyzing the results

To compute the results from the experiments, run the analysis scripts in `mof_benchmark/analysis`. You can quickly run everything using:


```bash
./run_analysis.sh
```

Results can then be plotted using the `plots.ipynb` notebook in `mof_benchmark/analysis/plot`.

Additionally, a Streamlit app is available to explore the results in depth.

Run it using:

```bash
cd mof_benchmark/analysis
streamlit run Overview.py
```


## :handshake: Contributing

The benchmark can be extended with new tasks and models due to its modular design.

To create a new task, we refer contributors to the interaction energy task in `mof_benchmark/experiments/scripts/interaction_energy` for a simple example.

Task classes are inherited from the `TaskRunner` which handles three aspects:
- Preparing calculator and structures based on the provided settings.
- Calling the task for each structure.
- Creation and cleaning of a temporary working directory, reducing filesystem load on distributed systems.

To perform a task, the `run_task` method has to be implemented. Storing results needs to be handled in this method as well.

Structures can be defined using file paths or loaded from the structure shortcuts defined in `mof_benchmark/experiments/structures/structures.yaml`.


## Citation

If you use this benchmark in your research, please cite our paper:

```bibtex
@misc{krass2025mofsimbench,
      title={MOFSimBench: Evaluating Universal Machine Learning Interatomic Potentials In Metal-Organic Framework Molecular Modeling}, 
      author={Hendrik Kraß and Ju Huang and Seyed Mohamad Moosavi},
      year={2025},
      eprint={2507.11806},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2507.11806},
}
```