import torch
import os
import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ggsp.metrics import p_losses

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
    checkpoint_path: str = None,
    device: Union[str, torch.device] = "cpu",
    verbose: bool = True,
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
        checkpoint_path (str, optional): path to save the best model. Defaults to None.
        device (Union[str, torch.device], optional): device. Defaults to "cpu".
        verbose (bool, optional): If True, print epochs. Defaults to True.

    Returns:
        pd.DataFrame: dataframe with train and validation metrics
    """
    df_metrics = pd.DataFrame(
        columns=[
                "datetime",
                "epoch",
                "train_huber_loss",
                "val_huber_loss",
            ]
    )

    # define alphas
    alphas = 1.0 - beta_schedule
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    best_val_loss = np.inf
    for epoch in range(1, epoch_number + 1):
        model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, diffusion_timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(
                model,
                x_g,
                t,
                data.stats,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                loss_type="huber",
            )
            loss.backward()
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()

        model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_dataloader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, diffusion_timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(
                model,
                x_g,
                t,
                data.stats,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                loss_type="huber",
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
                        "train_huber_loss": train_loss_all / train_count,
                        "val_huber_loss": val_loss_all / val_count,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        ).reset_index(drop=True)
        
        if verbose:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(
                "{} Epoch: {:04d}/{:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}".format(
                    df_metrics.iloc[-1]["datetime"],
                    df_metrics.iloc[-1]["epoch"],
                    epoch_number,
                    df_metrics.iloc[-1]["train_huber_loss"],
                    df_metrics.iloc[-1]["val_huber_loss"],
                )
            )

        scheduler.step()

        if best_val_loss >= val_loss_all and checkpoint_path is not None:
            best_val_loss = val_loss_all
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_path
            )

    return df_metrics