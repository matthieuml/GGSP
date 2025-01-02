import torch
import logging
import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ggsp.metrics import p_losses

logger = logging.getLogger("GGSP")


def train_denoiser(
    model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epoch_number: int,
    diffusion_timesteps: int,
    beta_schedule: torch.Tensor,
    loss_type: str = "huber",
    checkpoint_path: str = None,
    device: Union[str, torch.device] = "cpu",
) -> pd.DataFrame:
    """Train denoiser model.

    Args:
        model (torch.nn.Module): denoiser model to train
        autoencoder (torch.nn.Module): autoencoder model to project data to latent space
        train_dataloader (DataLoader): training dataloader
        val_dataloader (DataLoader): validation dataloader
        optimizer (Optimizer): learning rate optimizer
        scheduler (_LRScheduler): learning rate scheduler
        epoch_number (int): number of epochs
        diffusion_timesteps (int): number of diffusion timesteps
        beta_schedule (torch.Tensor): noising beta schedule
        loss_type (str, optional): loss type. Defaults to "huber".
        checkpoint_path (str, optional): path to save the best model. Defaults to None.
        device (Union[str, torch.device], optional): device. Defaults to "cpu".
        verbose (bool, optional): If True, print epochs. Defaults to True.

    Returns:
        pd.DataFrame: dataframe with train and validation metrics
    """
    logger.info(f"Training {model.__class__.__name__} model over {epoch_number} epochs")
    logger.debug(f"Using {loss_type} loss")

    df_metrics = pd.DataFrame(
        columns=[
            "datetime",
            "epoch",
            f"train_{loss_type}_loss",
            f"val_{loss_type}_loss",
        ]
    )

    # define alphas
    alphas = 1.0 - beta_schedule
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    best_val_loss = np.inf
    previous_lr = scheduler.get_last_lr()
    for epoch in range(1, epoch_number + 1):
        logger.debug(f"Epoch: {epoch}, switching model to train mode")
        model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(
                0, diffusion_timesteps, (x_g.size(0),), device=device
            ).long()
            loss = p_losses(
                model,
                x_g,
                t,
                data.stats,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                loss_type=loss_type,
            )
            loss.backward()
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()

        logger.debug(f"Epoch: {epoch}, switching model to eval mode")
        model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_dataloader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(
                0, diffusion_timesteps, (x_g.size(0),), device=device
            ).long()
            loss = p_losses(
                model,
                x_g,
                t,
                data.stats,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                loss_type=loss_type,
            )
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

        df_metrics = pd.concat(
            [
                df_metrics,
                pd.DataFrame(
                    {
                        "datetime": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        "epoch": epoch,
                        f"train_{loss_type}_loss": train_loss_all / train_count,
                        f"val_{loss_type}_loss": val_loss_all / val_count,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        ).reset_index(drop=True)

        logger.info(
            "Epoch: {:04d}/{:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}".format(
                df_metrics.iloc[-1]["epoch"],
                epoch_number,
                df_metrics.iloc[-1][f"train_{loss_type}_loss"],
                df_metrics.iloc[-1][f"val_{loss_type}_loss"],
            )
        )

        scheduler.step()
        if not np.allclose(scheduler.get_last_lr(), previous_lr, atol=0):
            previous_lr = scheduler.get_last_lr()
            logger.debug(f"Learning rate changed to {previous_lr}")

        if best_val_loss >= val_loss_all and checkpoint_path is not None:
            logger.debug(
                f"New best checkpoint found at epoch {epoch}, saving to {checkpoint_path}"
            )
            best_val_loss = val_loss_all
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_path,
            )

    return df_metrics
