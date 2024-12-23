import torch
import os
import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def train_autoencoder(
    model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epoch_number: int,
    device: Union[str, torch.device] = "cpu",
    verbose: bool = True,
) -> pd.DataFrame:

    best_val_loss = np.inf
    for epoch in range(1, args.epochs_denoise + 1):
        model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
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

        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(
                denoise_model,
                x_g,
                t,
                data.stats,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                loss_type="huber",
            )
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(
                "{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}".format(
                    dt_t, epoch, train_loss_all / train_count, val_loss_all / val_count
                )
            )

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save(
                {
                    "state_dict": denoise_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                "denoise_model.pth.tar",
            )