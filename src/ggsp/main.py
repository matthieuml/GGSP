import argparse
import os
import torch
import sys

script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
print(project_root)
sys.path.append(project_root)

from ggsp.utils import load_yaml_into_namespace, make_dirs, set_seed, copy_file
from ggsp.runners import run_experiment


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "base.yaml"),
        help="Path to the config file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args = load_yaml_into_namespace(args.config, args)
    
    # Paths to save the results
    args.exp_path, args.checkpoints_path, args.visualizations_path = make_dirs(
        args.exp_path
    )
    args.vae_save_checkpoint_path = os.path.join(
        args.checkpoints_path, "best_autoencoder_checkpoint.pth.tar"
    ) if args.vae_save_checkpoint else None
    args.denoise_save_checkpoint_path = os.path.join(
        args.checkpoints_path, "best_denoiser_checkpoint.pth.tar"
    ) if args.denoise_save_checkpoint else None
    args.denoise_metrics_path = os.path.join(args.exp_path, "train_denoise_metrics.csv")
    args.vae_metrics_path = os.path.join(args.exp_path, "train_vae_metrics.csv")
    args.submission_file_path = os.path.join(args.exp_path, "submission.csv")

    # Save the config file in the experiment folder
    copy_file(args.config, os.path.join(args.exp_path, "config.yaml"))

    # execute the experiment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    run_experiment(args, device)
