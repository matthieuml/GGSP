import argparse
import torch
import logging
from torch_geometric.loader import DataLoader
from typing import Union

from ggsp.data import *
from ggsp.models import VariationalAutoEncoder, DenoiseNN
from ggsp.train import train_autoencoder, train_denoiser
from ggsp.utils import load_model_checkpoint
from ggsp.utils.noising_schedule import *
from ggsp.metrics import graph_norm_from_adj

import optuna

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("GGSP")


def run_grid_search(args: argparse.Namespace, device: Union[str, torch.device]) -> None:
    """Run the experiment with the NeuralGraphGenerator model.

    Args:
        args (argparse.Namespace): arguments of the experiment
        device (Union[str, torch.device]): device to run the experiment on
    """

    # Load the dataset
    trainset = globals()[args.dataset_preprocessing_function](
        args.dataset_folder, args.training_dataset, args.n_max_nodes, args.spectral_emb_dim
    )
    validset = globals()[args.dataset_preprocessing_function](
        args.dataset_folder, args.valid_dataset, args.n_max_nodes, args.spectral_emb_dim
    )

    # initialize VGAE dataloader
    train_loader_autoencoder = DataLoader(
    trainset, batch_size=args.batch_size_autoencoder, shuffle=args.shuffle_train
    )
    val_loader_autoencoder = DataLoader(
        validset, batch_size=args.batch_size_autoencoder, shuffle=args.shuffle_val
    )

    # initialize denoiser dataloader
    train_loader_denoise = DataLoader(
    trainset, batch_size=args.batch_size_denoise, shuffle=args.shuffle_train
    )
    val_loader_denoise = DataLoader(
        validset, batch_size=args.batch_size_denoise, shuffle=args.shuffle_val
    )

    

    logger.info(f"Train set size: {len(trainset)}")
    logger.info(f"Validation set size: {len(validset)}")

    def objective_ggsp(trial):
        # Sample hyperparameters from the trial
        # args.hidden_dim_encoder = 144
        # args.hidden_dim_decoder = 353
        # args.latent_dim = trial.suggest_int("latent_dim", 2, 64)
        args.n_layers_encoder = trial.suggest_int("n_layers_encoder", 2, 10)
        args.n_layers_decoder = trial.suggest_int("n_layers_decoder", 2, 10)
        # args.contrastive_loss_k = trial.suggest_int("contrastive_loss_k", 0, 5)
        # args.epochs_autoencoder = 50
        # args.epochs_denoiser = 30


        # initialize VGAE model
        autoencoder = VariationalAutoEncoder(
        args.spectral_emb_dim + 1,
        args.hidden_dim_encoder,
        args.hidden_dim_decoder,
        args.latent_dim,
        args.n_layers_encoder,
        args.n_layers_decoder,
        args.n_max_nodes,
        args.encoder_classname,
        args.decoder_classname,
        args.vae_kld_weight,
        args.vae_contrastive_weight,
        ).to(device)

        vae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.vae_lr)
        vae_scheduler = torch.optim.lr_scheduler.StepLR(
            vae_optimizer,
            step_size=args.vae_scheduler_step_size,
            gamma=args.vae_scheduler_gamma,
        )

        # Train VGAE model
        if args.vae_load_checkpoint_path is not None:
            load_model_checkpoint(autoencoder, vae_optimizer, args.vae_load_checkpoint_path, device)

        if args.train_autoencoder:
            vae_metrics = train_autoencoder(
                model=autoencoder,
                train_dataloader=train_loader_autoencoder,
                val_dataloader=val_loader_autoencoder,
                optimizer=vae_optimizer,
                scheduler=vae_scheduler,
                epoch_number=args.epochs_autoencoder,
                device=device,
                checkpoint_path=args.vae_save_checkpoint_path,
                kld_weight=args.vae_kld_weight,
                is_kld_weight_adaptative=args.is_kld_weight_adaptative,
                contrastive_loss_k=args.contrastive_loss_k,
                vae_temperature_contrastive=args.vae_temperature_contrastive,
            )
            vae_metrics.to_csv(args.vae_metrics_path, index=False)

            logger.debug("VAE Training finished")

        logger.debug(f"Switching {autoencoder.__class__.__name__} model to eval mode")
        autoencoder.eval()

        graph_losses = torch.tensor([])
        for data in val_loader_autoencoder:
            data = data.to(device)
            adj = autoencoder(data)
            graph_losses = torch.cat(
                (graph_losses, graph_norm_from_adj(adj.detach().cpu().numpy(), data.A.detach().cpu().numpy(), norm_type=args.graph_metric))
            )
        
        return (graph_losses.mean().item() + graph_losses.std().item()/2) / 256

        # define beta schedule
        logger.debug(f"Using {args.noising_schedule_function} function as noising schedule")
        betas = globals()[args.noising_schedule_function](timesteps=args.timesteps)

        # initialize denoising model
        denoise_model = DenoiseNN(
            input_dim=args.latent_dim,
            hidden_dim=args.hidden_dim_denoise,
            n_layers=args.n_layers_denoise,
            n_cond=args.n_condition,
            d_cond=args.dim_condition,
        ).to(device)

        denoise_optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.denoise_lr)
        denoise_scheduler = torch.optim.lr_scheduler.StepLR(
            denoise_optimizer,
            step_size=args.denoise_scheduler_step_size,
            gamma=args.vae_scheduler_gamma,
        )

        if args.denoise_load_checkpoint_path is not None:
            load_model_checkpoint(
                denoise_model, denoise_optimizer, args.denoise_load_checkpoint_path
            )

        # Train denoising model
        if args.train_denoise:
            denoise_metrics = train_denoiser(
                model=denoise_model,
                autoencoder=autoencoder,
                train_dataloader=train_loader_denoise,
                val_dataloader=val_loader_denoise,
                optimizer=denoise_optimizer,
                scheduler=denoise_scheduler,
                epoch_number=args.epochs_denoise,
                diffusion_timesteps=args.timesteps,
                beta_schedule=betas,
                loss_type=args.denoise_loss_type,
                device=device,
                checkpoint_path=args.denoise_save_checkpoint_path,
            )
            denoise_metrics.to_csv(args.denoise_metrics_path, index=False)

        denoise_model.eval()

        graph_losses = torch.tensor([])
        for data in val_loader_autoencoder:
            data = data.to(device)
            adj = autoencoder(data)
            graph_losses = torch.cat(
                (graph_losses, graph_norm_from_adj(adj.detach().cpu().numpy(), data.A.detach().cpu().numpy(), norm_type=args.graph_metric))
            )

        del autoencoder, denoise_model

        return (graph_losses.mean().item() + graph_losses.std().item()/2) / 256

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_ggsp, n_trials=500)

    # Get best hyperparameters
    best_trial = study.best_trial
    print(f"Best trial: {best_trial}")
    print(f"Best hyperparameters: {best_trial.params}")
