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
from ggsp.runners import generate_submission
from ggsp.metrics import absolute_loss_features
from ggsp.models import sample
import numpy as np

import optuna

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("GGSP")

EPOCHS_AUTOENCODER = 200
EPOCHS_DENOISER = 100


def run_grid_search(args: argparse.Namespace, device: Union[str, torch.device]) -> None:
    """Run the experiment with the NeuralGraphGenerator model.

    Args:
        args (argparse.Namespace): arguments of the experiment
        device (Union[str, torch.device]): device to run the experiment on
    """

    # Load the dataset
    trainset = globals()[args.dataset_preprocessing_function](
        args.dataset_folder, "train", args.n_max_nodes, args.spectral_emb_dim
    )
    validset = globals()[args.dataset_preprocessing_function](
        args.dataset_folder, "valid", args.n_max_nodes, args.spectral_emb_dim
    )

    TRAIN_LOADER = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=args.shuffle_train
    )
    VAL_LOADER = DataLoader(
        validset, batch_size=args.batch_size, shuffle=args.shuffle_val
    )

    logger.info(f"Train set size: {len(trainset)}")
    logger.info(f"Validation set size: {len(validset)}")

    def objective_ggsp(trial):

        # # Sample hyperparameters from the trial
        # spectral_emb_dim = 11
        # hidden_dim_encoder = trial.suggest_int('hidden_dim_encoder', 64, 511)
        # latent_dim = trial.suggest_int('latent_dim', 16, 256)
        # hidden_dim_decoder = trial.suggest_int('hidden_dim_decoder', 64, 511)
        # n_layers_encoder = trial.suggest_int('n_layers_encoder', 2, 5)
        # n_layers_decoder = trial.suggest_int('n_layers_decoder', 2, 5)
        # n_max_nodes = 50
        # encoder_classname = trial.suggest_categorical('encoder_classname', ['GIN'])
        # decoder_classname = trial.suggest_categorical('decoder_classname', ['Decoder'])

        # Sample hyperparameters from the trial
        spectral_emb_dim = 11
        hidden_dim_encoder = trial.suggest_int("hidden_dim_encoder", 32, 128) #256
        latent_dim = 19
        hidden_dim_decoder = trial.suggest_int("hidden_dim_decoder", 400, 600) #512
        n_layers_encoder = trial.suggest_int("n_layers_encoder", 3, 10) #2
        n_layers_decoder = trial.suggest_int("n_layers_decoder", 3, 10) #3
        n_max_nodes = 50
        beta = 3.5e-07
        encoder_classname = trial.suggest_categorical('encoder_classname', ['GIN'])
        decoder_classname = trial.suggest_categorical('decoder_classname', ['Decoder'])
        epochs_autoencoder = trial.suggest_int("epochs_autoencoder", 50, 300) #200
        epochs_denoiser = trial.suggest_int("epochs_denoiser", 50, 300) #100

        print(f"Spec dim: {spectral_emb_dim}, hidden_dim_encoder: {hidden_dim_encoder}, hidden_dim_decoder: {hidden_dim_decoder}, latent_dim: {latent_dim}, n_layers_encoder: {n_layers_encoder}, n_layers_decoder: {n_layers_decoder}, n_max_nodes: {n_max_nodes}, beta: {beta}, encoder_classname: {encoder_classname}, decoder_classname: {decoder_classname}, epochs_autoencoder: {epochs_autoencoder},
epochs_denoiser: {epochs_denoiser}")

        # initialize VGAE model
        autoencoder = VariationalAutoEncoder(
            spectral_emb_dim,
            hidden_dim_encoder,
            hidden_dim_decoder,
            latent_dim,
            n_layers_encoder,
            n_layers_decoder,
            n_max_nodes,
            encoder_classname,
            decoder_classname,
            beta
        ).to(device)

        vae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.vae_lr)
        vae_scheduler = torch.optim.lr_scheduler.StepLR(
            vae_optimizer,
            step_size=args.vae_scheduler_step_size,
            gamma=args.vae_scheduler_gamma,
        )

        # Train VGAE model
        if args.vae_load_checkpoint_path is not None:
            load_model_checkpoint(autoencoder, vae_optimizer, args.vae_load_checkpoint_path)

        if args.train_autoencoder:
            vae_metrics = train_autoencoder(
                model=autoencoder,
                train_dataloader=TRAIN_LOADER,
                val_dataloader=VAL_LOADER,
                optimizer=vae_optimizer,
                scheduler=vae_scheduler,
                epoch_number=epochs_autoencoder,
                device=device,
                checkpoint_path=args.vae_save_checkpoint_path,
            )
            vae_metrics.to_csv(args.vae_metrics_path, index=False)

            logger.debug("VAE Training finished")

        logger.debug(f"Switching {autoencoder.__class__.__name__} model to eval mode")
        autoencoder.eval()

        # define beta schedule
        logger.debug(f"Using {args.noising_schedule_function} function as noising schedule")
        betas = globals()[args.noising_schedule_function](timesteps=args.timesteps)

        # initialize denoising model
        denoise_model = DenoiseNN(
            input_dim=latent_dim,
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
                train_dataloader=TRAIN_LOADER,
                val_dataloader=VAL_LOADER,
                optimizer=denoise_optimizer,
                scheduler=denoise_scheduler,
                epoch_number=epochs_denoiser,
                diffusion_timesteps=args.timesteps,
                beta_schedule=betas,
                loss_type=args.denoise_loss_type,
                device=device,
                checkpoint_path=args.denoise_save_checkpoint_path,
            )
            denoise_metrics.to_csv(args.denoise_metrics_path, index=False)

        denoise_model.eval()

        loss_val = []
        for data in VAL_LOADER:
            data = data.to(device)
            samples = sample(
                    denoise_model,
                    data.stats,
                    latent_dim=latent_dim,
                    timesteps=args.timesteps,
                    betas=betas,
                    batch_size=data.stats.size(0)
                )
            x_sample = samples[-1]
            adj = autoencoder.decode_mu(x_sample)
            loss_val.append(absolute_loss_features(adj, data.A, data).sum().item())

        return round(np.sum(loss_val)/len(validset), 2)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_ggsp, n_trials=500)

    # Get best hyperparameters
    best_trial = study.best_trial
    print(f"Best trial: {best_trial}")
    print(f"Best hyperparameters: {best_trial.params}")
