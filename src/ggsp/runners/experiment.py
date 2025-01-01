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
from ggsp.metrics import graph_norm_from_adj
from ggsp.models import sample

logger = logging.getLogger("GGSP")


def run_experiment(args: argparse.Namespace, device: Union[str, torch.device]) -> None:
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
    testset = globals()[args.dataset_preprocessing_function](
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

    logger.info(f"Train set size: {len(trainset)}")
    logger.info(f"Validation set size: {len(validset)}")
    logger.info(f"Test set size: {len(testset)}")

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
            checkpoint_path=args.vae_save_checkpoint_path,
            kld_weight=args.vae_kld_weight,
        )
        vae_metrics.to_csv(args.vae_metrics_path, index=False)

        logger.debug(f"VAE Training finished")

    logger.debug(f"Switching {autoencoder.__class__.__name__} model to eval mode")
    autoencoder.eval()

    if args.graph_metric is not None:
        graph_losses = torch.tensor([])
        for data in tqdm(
            val_loader,
            desc="Computing graph metric on validation set",   
        ):
            data = data.to(device)
            adj = autoencoder(data)
            graph_losses = torch.cat(
                (graph_losses, graph_norm_from_adj(adj.detach().cpu().numpy(), data.A.detach().cpu().numpy(), norm_type=args.graph_metric))
            )
        
        # TODO : Remove the division by batch size, just to fit to kaggle results
        logger.info(f"{args.graph_metric} loss on VAE: {graph_losses.mean().item() / args.batch_size}")

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
            train_dataloader=train_loader,
            val_dataloader=val_loader,
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
    if args.graph_metric is not None and args.train_denoise:
        graph_losses = torch.tensor([])
        for data in tqdm(
            val_loader,
            desc="Computing graph metric on validation set",   
        ):
            data = data.to(device)
            # Sample and denoise the data base on conditioning
            samples = sample(
                denoise_model,
                data.stats,
                latent_dim=args.latent_dim,
                timesteps=args.timesteps,
                betas=betas,
                batch_size=data.stats.size(0),
            )
            # Take the last sample as the denoised sample and decode it
            x_sample = samples[-1]
            adj = autoencoder.decode_mu(x_sample)
            graph_losses = torch.cat(
                (graph_losses, graph_norm_from_adj(adj.detach().cpu().numpy(), data.A.detach().cpu().numpy(), norm_type=args.graph_metric))
            )

        # TODO : Remove the division by batch size, just to fit to kaggle results
        logger.info(f"{args.graph_metric} loss on global pipeline: {graph_losses.mean().item() / args.batch_size}")


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
