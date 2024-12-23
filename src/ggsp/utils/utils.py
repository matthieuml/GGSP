import math
import torch
from torch.optim.optimizer import Optimizer

def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x

def load_model_checkpoint(model: torch.nn.Module, optimizer: Optimizer, checkpoint_path: str):
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

