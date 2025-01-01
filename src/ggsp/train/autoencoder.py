import torch
import logging
import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger("GGSP")


def train_autoencoder(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epoch_number: int,
    kld_weight: float = 0.05,
    checkpoint_path: str = None,
    device: Union[str, torch.device] = "cpu",
) -> pd.DataFrame:
    """Train autoencoder model.

    Args:
        model (torch.nn.Module): autoencoder model to train
        train_dataloader (DataLoader): training dataloader
        val_dataloader (DataLoader): validation dataloader
        optimizer (Optimizer): learning rate optimizer
        scheduler (_LRScheduler): learning rate scheduler
        epoch_number (int): number of epochs
        kld_weight (float, optional): weight of the KLD loss. Defaults to 0.05.
        checkpoint_path (str, optional): path to save the best model. Defaults to None.
        device (Union[str, torch.device], optional): device. Defaults to "cpu".
        verbose (bool, optional): If True, print epochs. Defaults to True.

    Returns:
        pd.DataFrame: dataframe with train and validation metrics
    """
    logger.info(f"Training {model.__class__.__name__} model over {epoch_number} epochs")
    df_metrics = pd.DataFrame(
        columns=[
            "datetime",
            "epoch",
            "train_loss",
            "train_reconstruction_loss",
            "train_kld_loss",
            "val_loss",
            "val_reconstruction_loss",
            "val_kld_loss",
        ]
    )

    best_val_loss = np.inf
    for epoch in range(1, epoch_number + 1):
        logger.debug(f"Epoch: {epoch}, switching model to train mode")
        model.train()
        train_loss_all = 0
        train_count = 0
        train_loss_all_recon = 0
        train_loss_all_kld = 0
        cnt_train = 0

        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, recon, kld = model.loss_function(data, beta=kld_weight)
            train_loss_all_recon += recon.item()
            train_loss_all_kld += kld.item()
            cnt_train += 1
            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch) + 1
            optimizer.step()

        logger.debug(f"Epoch: {epoch}, switching model to eval mode")
        model.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_all_recon = 0
        val_loss_all_kld = 0

        for data in val_dataloader:
            data = data.to(device)
            loss, recon, kld = model.loss_function(data)
            val_loss_all_recon += recon.item()
            val_loss_all_kld += kld.item()
            val_loss_all += loss.item()
            cnt_val += 1
            val_count += torch.max(data.batch) + 1

        df_metrics = pd.concat(
            [
                df_metrics,
                pd.DataFrame(
                    {
                        "datetime": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        "epoch": epoch,
                        "train_loss": train_loss_all / cnt_train,
                        "train_reconstruction_loss": train_loss_all_recon / cnt_train,
                        "train_kld_loss": train_loss_all_kld / cnt_train,
                        "val_loss": val_loss_all / cnt_val,
                        "val_reconstruction_loss": val_loss_all_recon / cnt_val,
                        "val_kld_loss": val_loss_all_kld / cnt_val,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        ).reset_index(drop=True)

        logger.info(
            "Epoch: {:04d}/{:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}".format(
                df_metrics.iloc[-1]["epoch"],
                epoch_number,
                df_metrics.iloc[-1]["train_loss"],
                df_metrics.iloc[-1]["train_reconstruction_loss"],
                df_metrics.iloc[-1]["train_kld_loss"],
                df_metrics.iloc[-1]["val_loss"],
                df_metrics.iloc[-1]["val_reconstruction_loss"],
                df_metrics.iloc[-1]["val_kld_loss"],
            )
        )

        scheduler.step()

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
