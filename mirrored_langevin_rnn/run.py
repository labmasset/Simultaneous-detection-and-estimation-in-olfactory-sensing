"""Command line entry point for running experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from mirrored_langevin_rnn.simulator.sensory_scene import sensing_matrix
from .config_loader import load_config
from .experiment import create_experiment
from .simulator.factory import create_simulator
from .logging_utils import setup_logging


def main() -> None:
    """
    Main entry point for running experiments or simulations.
    """
    parser = argparse.ArgumentParser(
        description="Run experiments or simulations from YAML configs"
    )
    parser.add_argument("config", type=str, help="Path to configuration YAML")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Optional path to save simulation results as NPZ. "
            "If omitted, results will be stored under ~/data using the "
            "config file name."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "serial", "parallel", "slurm"],
        default="auto",
        help="Execution mode for experiments: auto (default), serial, parallel, or slurm"
    )
    args = parser.parse_args()

    kind, system_cfg, payload = load_config(args.config)
    # payload is now a parameter dataclass instance

    setup_logging(system_cfg.log_level, system_cfg.log_file)

    device = system_cfg.apply()
    payload.device = device
    
    """
    We now build and run the experiment or simulator 
    based on the parameter dataclass "payload".
    Creating of the experiment or simulator 
    is done by the factory functions in 
    `experiment` or `simulator` modules.
    """
    if kind == "experiment":
        exp = create_experiment(payload)
        exp.run(mode=args.mode)
    else:
        sim = create_simulator(payload)

        if args.out is None:
            # Use project's data directory instead of home directory
            project_root = Path(__file__).parent.parent
            sensing_matrix_type = payload.sensing_matrix_type
            num_low = payload.num_low
            if num_low > 0:
                default_dir = project_root / "data" / "simulations" / 'steps' / sensing_matrix_type
            else:
                default_dir = project_root / "data" / "simulations" / 'static' / sensing_matrix_type
            default_dir.mkdir(parents=True, exist_ok=True)
            default_path = default_dir / f"{Path(args.config).stem}.npz"
        else:
            default_path = Path(args.out)

        sim.simulate(include_ground_truth=True, save_path=str(default_path))


if __name__ == "__main__":
    main()
