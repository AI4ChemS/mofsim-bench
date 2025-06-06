import logging
import sys
import yaml
from ase.calculators.calculator import Calculator
import os
from pathlib import Path
from ase import units

# Set up logger
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_calculator(model: str) -> Calculator:
    """
    Get the calculator object for the MD simulation.

    Parameters
    ----------
    model : str
        Name of the model to use.

    Returns
    -------
    ase.Calculator
        Calculator object for the MD simulation
    """

    local_dir = Path(os.path.dirname(__file__))
    settings = yaml.safe_load(open(f"{local_dir}/calculators.yaml"))

    if model not in settings:
        logger.error(f"Model {model} not found in calculators.yaml")
        logger.error(f"Available models: {list(settings.keys())}")
        sys.exit(1)

    settings = settings[model]

    if "mace" in model:
        from mace.calculators import mace_mp
        import torch as th

        device = settings.get("device", "cuda")
        if device == "cuda":
            if not th.cuda.is_available():
                logger.error("CUDA not available, using CPU")
                device = "cpu"
        model_file = settings.get(
            "model_file",
            "../../../mace4mof/mace4mof/model_management/2024-07-12-mace-128-L1_epoch-199.model",
        )
        enable_cueq = settings.get("enable_cueq", False)
        precision = settings.get("precision", "float32")
        with_d3 = settings.get("with_d3", False)
        dispersion_xc = settings.get("dispersion_xc", "pbe")
        dispersion_cutoff = settings.get("dispersion_cutoff", 40.0 * units.Bohr)
        damping = settings.get("damping", "bj")

        logger.info(f"Using model {model}")
        logger.info(f"Using model {model_file}")
        logger.info(f"Using device {device}")
        logger.info(f"Using precision {precision}")

        calc = mace_mp(
            model=os.path.join(local_dir, model_file),
            device=device,
            default_dtype=precision,
            enable_cueq=enable_cueq,
        )

    if "orb" in model:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS
        import torch as th

        device = settings.get("device", "cuda")
        if device == "cuda":
            if not th.cuda.is_available():
                logger.error("CUDA not available, using CPU")
                device = "cpu"

        model_name = settings.get("model_name", "orb-v2")
        model_kwargs = settings.get("model_kwargs", {})
        orbff = ORB_PRETRAINED_MODELS[model_name](**model_kwargs)
        calc = ORBCalculator(orbff, device=device)

        logger.info(f"Using ORB model {model_name}")
        logger.info(f"Using device {device}")

    if "omat24" in model:
        from fairchem.core import OCPCalculator
        import torch as th

        device = settings.get("device", "cuda")
        if device == "cuda":
            if not th.cuda.is_available():
                logger.error("CUDA not available, using CPU")
                device = "cpu"

        checkpoint_path = settings.get("checkpoint_path", "eqV2_86M_omat_mp_salex.pt")

        calc = OCPCalculator(
            checkpoint_path=local_dir / checkpoint_path,
            local_cache="pretrained_models",
            cpu=device == "cpu",
            seed=0,
        )

        logger.info(f"Using OMAT24 model {checkpoint_path}")
        logger.info(f"Using device {device}")

    if "grace" in model:

        from tensorpotential.calculator import grace_fm

        model_name = settings.get("model_name", "MP_GRACE_2L_r6_11Nov2024")
        calc = grace_fm(model_name)

        logger.info(f"Using GRACE model {model_name}")

        device = "cuda"

    if "mattersim" in model:

        from mattersim.forcefield import MatterSimCalculator
        import torch as th

        device = settings.get("device", "cuda")
        if device == "cuda":
            if not th.cuda.is_available():
                logger.error("CUDA not available, using CPU")
                device = "cpu"

        load_path = settings.get("load_path", "MatterSim-v1.0.0-5M.pth")

        calc = MatterSimCalculator(load_path=load_path, device=device)

        logger.info(f"Using MatterSim model {load_path}")
        logger.info(f"Using device {device}")

    if "sevennet" in model:

        from sevenn.sevennet_calculator import SevenNetCalculator
        import torch as th

        device = settings.get("device", "cuda")
        if device == "cuda":
            if not th.cuda.is_available():
                logger.error("CUDA not available, using CPU")
                device = "cpu"

        model_name = settings.get("model_name", "7net-l3i5")
        kwargs = settings.get("kwargs", {})
        calc = SevenNetCalculator(model=model_name, device=device, **kwargs)

        logger.info(f"Using SevenNet model {model_name}")
        logger.info(f"Using device {device}")

    if "posegnn" in model:
        from posegnn.calculator import PosEGNNCalculator
        import torch as th

        device = settings.get("device", "cuda")
        th.set_float32_matmul_precision("high")

        if device == "cuda":
            if not th.cuda.is_available():
                logger.error("CUDA not available, using CPU")
                device = "cpu"

        checkpoint = settings.get("checkpoint", "pos-egnn.v1-6M.ckpt")
        calc = PosEGNNCalculator(
            os.path.join(local_dir, checkpoint), device=device, compute_stress=True
        )

        logger.info(f"Using PosEGNN model {checkpoint}")
        logger.info(f"Using device {device}")

    if "matgl" in model:
        import matgl
        from matgl.ext.ase import PESCalculator

        model_name = settings.get("model_name", None)
        calc = PESCalculator(matgl.load_model(model_name))

    with_d3 = settings.get("with_d3", False)
    dispersion_xc = settings.get("dispersion_xc", "pbe")
    dispersion_cutoff = settings.get("dispersion_cutoff", 40.0 * units.Bohr)
    damping = settings.get("damping", "bj")

    if with_d3:

        import torch as th

        device = settings.get("device", "cuda")
        if device == "cuda":
            if not th.cuda.is_available():
                logger.error("CUDA not available, using CPU")
                device = "cpu"

        from ase.calculators.mixing import SumCalculator
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

        logger.info(f"Using dispersion XC {dispersion_xc}")
        logger.info(f"Using dispersion cutoff {dispersion_cutoff}")
        logger.info(f"Using damping {damping}")

        calc = SumCalculator(
            calcs=[
                calc,
                TorchDFTD3Calculator(
                    device=device,
                    damping=damping,
                    xc=dispersion_xc,
                    cutoff=dispersion_cutoff,
                ),
            ]
        )

    return calc
