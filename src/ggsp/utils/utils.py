import math
import torch
import argparse
import os
import random
import yaml
import numpy as np
from torch.optim.optimizer import Optimizer


def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x


def load_model_checkpoint(
    model: torch.nn.Module, optimizer: Optimizer, checkpoint_path: str
):
    """Load model checkpoint.

    Args:
        model (torch.nn.Module): model to load
        optimizer (Optimizer): optimizer to load
        checkpoint_path (str): checkpoint path
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def load_yaml_into_namespace(
    yaml_file: str, namespace: argparse.Namespace
) -> argparse.Namespace:
    """
    Load a YAML file and merge its content into the given argparse Namespace.

    Args:
        yaml_file (str): Path to the YAML file.
        namespace (argparse.Namespace): The current Namespace object.

    Returns:
        argparse.Namespace: Updated Namespace with the values from the YAML file.
    """
    # Load the YAML file
    with open(yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)  # Parse the YAML content as a dictionary

    # Merge the YAML data into the Namespace
    namespace_dict = vars(namespace)  # Convert Namespace to dictionary
    namespace_dict.update(yaml_data)  # Update with YAML data
    return argparse.Namespace(**namespace_dict)  # Convert back to Namespace


def make_dirs(
    experiment_path: str,
) -> tuple[str, str, str]:
    """Create directories for the experiment.

    Args:
        experiment_path (str): path to the experiment
    """
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    else:
        parent, folder_name = os.path.split(experiment_path)
        counter = 1
        while True:
            new_folder_name = f"{folder_name}_{counter}"
            experiment_path = os.path.join(parent, new_folder_name)
            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)
                break
            counter += 1

    check_point_path = os.path.join(experiment_path, "checkpoints")
    visualizations_path = os.path.join(experiment_path, "visuals")
    os.makedirs(check_point_path, exist_ok=True)
    os.makedirs(visualizations_path, exist_ok=True)

    return experiment_path, check_point_path, visualizations_path


def set_seed(seed: int) -> None:
    """
    Set the seed for Python's random module, numpy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    # Python's built-in random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments

    # Ensure deterministic behavior in PyTorch (if applicable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seeds set to {seed}")
