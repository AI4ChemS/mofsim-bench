import os
from pathlib import Path
from typing import List

import ase.io
import yaml

from mof_benchmark.experiments.scripts.utils import (
    logger,
    TmpWorkdir,
    get_path,
    structure_to_files,
    update_dicts,
)
from mof_benchmark.setup.calculator import get_calculator


class TaskRunner:
    def __init__(
        self,
        calculator: str | None = None,
        structure: str | List[str] | None = None,
        settings: str | dict = None,
        default_settings: dict = {},
        index: int | None = None,
        **kwargs,
    ):
        if not isinstance(settings, dict):
            try:
                with open(settings, "r") as f:
                    logger.info(f"Loading settings from {settings}")
                    settings = yaml.safe_load(f)
            except:
                settings = {}
        settings = update_dicts(default_settings, kwargs, (settings or {}))

        if calculator is None:
            calculator = settings.get("calculator")

        if structure is None:
            structure = settings.get("structure")

        assert calculator is not None, "Calculator not provided."
        assert structure is not None, "Structure not provided."

        self.calculator = calculator
        logger.info(f"Using calculator {calculator}")
        self.calc = get_calculator(calculator)
        self.files = structure_to_files(structure)
        if index is not None:
            self.files = [self.files[index]]
        self.settings = settings

    def task(self, atoms, settings, **kwargs):
        raise NotImplementedError

    def skip(self, **kwargs):
        return True

    def run(self):
        with TmpWorkdir(
            output_dir=get_path(
                self.settings["output_dir"], calculator=self.calculator
            ),
            tmp_dir=get_path(self.settings["tmp_dir"], calculator=self.calculator),
            copy_outputs=True,
        ):
            logger.info(f"Working in {os.getcwd()}.")
            logger.info(f"Starting tasks.")

            for file in self.files:

                try:

                    structure_name = Path(file).stem

                    kwargs = {
                        "structure_name": structure_name,
                        "calculator": self.calculator,
                    }

                    if self.settings.get("skip_existing", True):
                        if self.skip(**kwargs):
                            logger.info(f"Skipping {structure_name}")
                            continue

                    logger.info(f"Running task on {structure_name}")

                    atoms = ase.io.read(file)
                    atoms.calc = self.calc

                    self.task(atoms, self.settings, **kwargs)

                except Exception as e:
                    logger.error(f"Error in {file}")
                    print(e)
