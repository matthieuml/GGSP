import argparse
import torch
from torch_geometric.loader import DataLoader
from typing import Union

from ggsp.data import preprocess_dataset
from ggsp.models import VariationalAutoEncoder, DenoiseNN
from ggsp.train import train_autoencoder, train_denoiser
from ggsp.utils import load_model_checkpoint, linear_beta_schedule
from ggsp.runners import generate_submission


def run_experiment(args: argparse.Namespace, device: Union[str, torch.device]) -> None:
    """Run the experiment with the NeuralGraphGenerator model.

    Args:
        args (argparse.Namespace): arguments of the experiment
        device (Union[str, torch.device]): device to run the experiment on
    """

    # Load the dataset
    trainset = preprocess_dataset(
        args.dataset_folder, "train", args.n_max_nodes, args.spectral_emb_dim
    )
    validset = preprocess_dataset(
        args.dataset_folder, "valid", args.n_max_nodes, args.spectral_emb_dim
    )
    testset = preprocess_dataset(
        args.dataset_folder, "test", args.n_max_nodes, args.spectral_emb_dim
    )

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=args.shuffle_train
    )
    val_loader = DataLoader(
        validset, batch_size=args.batch_size, shuffle=args.shuffle_val
    )
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=args.shuffle_test
    )

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
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=vae_optimizer,
            scheduler=vae_scheduler,
            epoch_number=args.epochs_autoencoder,
            device=device,
            verbose=args.verbose,
            checkpoint_path=args.vae_save_checkpoint_path,
        )
        vae_metrics.to_csv(args.vae_metrics_path, index=False)

    autoencoder.eval()

    # define beta schedule
    betas = linear_beta_schedule(timesteps=args.timesteps)

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
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=denoise_optimizer,
            scheduler=denoise_scheduler,
            epoch_number=args.epochs_denoise,
            diffusion_timesteps=args.timesteps,
            beta_schedule=betas,
            loss_type=args.denoise_loss_type,
            device=device,
            verbose=args.verbose,
            checkpoint_path=args.denoise_save_checkpoint_path,
        )
        denoise_metrics.to_csv(args.denoise_metrics_path, index=False)

    denoise_model.eval()
    del train_loader, val_loader

    # Generate submission file on the test set
    if args.submission_file_path is not None:
        generate_submission(
            autoencoder=autoencoder,
            denoise_model=denoise_model,
            beta_schedule=betas,
            test_loader=test_loader,
            file_path=args.submission_file_path,
            args=args,
            device=device,
        )