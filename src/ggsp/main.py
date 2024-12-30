import argparse
import os
import torch
import sys

script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
print(project_root)
sys.path.append(project_root)

from ggsp.utils import (
    load_yaml_into_namespace,
    make_dirs,
    set_seed,
    copy_file,
    setup_logger,
)
from ggsp.runners import run_experiment, run_grid_search


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "base.yaml"),
        help="Path to the config file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="experiment",
        choices=["experiment", "grid_search"],
        help="Mode of the experiment",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    args = load_yaml_into_namespace(args.config, args)

    # Paths to save the results
    args.exp_path, args.checkpoints_path, args.visualizations_path = make_dirs(
        os.path.join(args.exp_path, args.exp_name)
    )
    args.vae_save_checkpoint_path = (
        os.path.join(args.checkpoints_path, "best_autoencoder_checkpoint.pth.tar")
        if args.vae_save_checkpoint
        else None
    )
    args.denoise_save_checkpoint_path = (
        os.path.join(args.checkpoints_path, "best_denoiser_checkpoint.pth.tar")
        if args.denoise_save_checkpoint
        else None
    )
    args.denoise_metrics_path = os.path.join(args.exp_path, "train_denoise_metrics.csv")
    args.vae_metrics_path = os.path.join(args.exp_path, "train_vae_metrics.csv")
    args.submission_file_path = (
        os.path.join(args.exp_path, "submission.csv") if args.submission_file else None
    )

    logger = setup_logger(
        name="GGSP",
        log_file=os.path.join(args.exp_path, f"{args.exp_name}.log")
        if args.save_logs
        else None,
        level=args.log_level,
        log_to_console=args.verbose,
    )

    # Save the config file in the experiment folder
    copy_file(args.config, os.path.join(args.exp_path, "config.yaml"))

    # execute the experiment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    set_seed(args.seed)
    if args.mode == "experiment":
        run_experiment(args, device)
    elif args.mode == "grid_search":
        run_grid_search(args, device)
    else:
        raise ValueError("Invalid mode")
    