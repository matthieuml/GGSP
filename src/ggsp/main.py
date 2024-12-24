import argparse
import os

import numpy as np
import torch

from torch_geometric.loader import DataLoader

from ggsp.models import VariationalAutoEncoder, DenoiseNN
from ggsp.train import train_autoencoder, train_denoiser
from ggsp.data import preprocess_dataset
from ggsp.utils import linear_beta_schedule, load_model_checkpoint
from ggsp.runners import generate_submission


np.random.seed(13)

"""
Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

# Argument parser
parser = argparse.ArgumentParser(description="NeuralGraphGenerator")

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(
    description="Configuration for the NeuralGraphGenerator model"
)

# Learning rate for the optimizer
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Learning rate for the optimizer, typically a small float value (default: 0.001)",
)

# Dropout rate
parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)",
)

# Batch size for training
parser.add_argument(
    "--batch-size",
    type=int,
    default=256,
    help="Batch size for training, controlling the number of samples per gradient update (default: 256)",
)

# Number of epochs for the autoencoder training
parser.add_argument(
    "--epochs-autoencoder",
    type=int,
    default=200,
    help="Number of training epochs for the autoencoder (default: 200)",
)

# Hidden dimension size for the encoder network
parser.add_argument(
    "--hidden-dim-encoder",
    type=int,
    default=64,
    help="Hidden dimension size for encoder layers (default: 64)",
)

# Hidden dimension size for the decoder network
parser.add_argument(
    "--hidden-dim-decoder",
    type=int,
    default=256,
    help="Hidden dimension size for decoder layers (default: 256)",
)

# Dimensionality of the latent space
parser.add_argument(
    "--latent-dim",
    type=int,
    default=32,
    help="Dimensionality of the latent space in the autoencoder (default: 32)",
)

# Maximum number of nodes of graphs
parser.add_argument(
    "--n-max-nodes",
    type=int,
    default=50,
    help="Possible maximum number of nodes in graphs (default: 50)",
)

# Number of layers in the encoder network
parser.add_argument(
    "--n-layers-encoder",
    type=int,
    default=2,
    help="Number of layers in the encoder network (default: 2)",
)

# Number of layers in the decoder network
parser.add_argument(
    "--n-layers-decoder",
    type=int,
    default=3,
    help="Number of layers in the decoder network (default: 3)",
)

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument(
    "--spectral-emb-dim",
    type=int,
    default=10,
    help="Dimensionality of spectral embeddings for representing graph structures (default: 10)",
)

# Number of training epochs for the denoising model
parser.add_argument(
    "--epochs-denoise",
    type=int,
    default=100,
    help="Number of training epochs for the denoising model (default: 100)",
)

# Number of timesteps in the diffusion
parser.add_argument(
    "--timesteps",
    type=int,
    default=500,
    help="Number of timesteps for the diffusion (default: 500)",
)

# Hidden dimension size for the denoising model
parser.add_argument(
    "--hidden-dim-denoise",
    type=int,
    default=512,
    help="Hidden dimension size for denoising model layers (default: 512)",
)

# Number of layers in the denoising model
parser.add_argument(
    "--n-layers_denoise",
    type=int,
    default=3,
    help="Number of layers in the denoising model (default: 3)",
)

# Flag to toggle training of the autoencoder (VGAE)
parser.add_argument(
    "--train-autoencoder",
    action="store_false",
    default=True,
    help="Flag to enable/disable autoencoder (VGAE) training (default: enabled)",
)

# Flag to toggle training of the diffusion-based denoising model
parser.add_argument(
    "--train-denoiser",
    action="store_true",
    default=True,
    help="Flag to enable/disable denoiser training (default: enabled)",
)

# Dimensionality of conditioning vectors for conditional generation
parser.add_argument(
    "--dim-condition",
    type=int,
    default=128,
    help="Dimensionality of conditioning vectors for conditional generation (default: 128)",
)

# Number of conditions used in conditional vector (number of properties)
parser.add_argument(
    "--n-condition",
    type=int,
    default=7,
    help="Number of distinct condition properties used in conditional vector (default: 7)",
)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
dataset_folder = os.path.join(os.path.dirname(__file__), "..", "..", "data")
trainset = preprocess_dataset(dataset_folder, "train", args.n_max_nodes, args.spectral_emb_dim)
validset = preprocess_dataset(dataset_folder, "valid", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_dataset(dataset_folder, "test", args.n_max_nodes, args.spectral_emb_dim)
print("Data loaded.")


# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)


# initialize VGAE model
autoencoder = VariationalAutoEncoder(
    args.spectral_emb_dim + 1,
    args.hidden_dim_encoder,
    args.hidden_dim_decoder,
    args.latent_dim,
    args.n_layers_encoder,
    args.n_layers_decoder,
    args.n_max_nodes,
).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)


# Train VGAE model
if args.train_autoencoder:
    train_autoencoder(
        model=autoencoder,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch_number=args.epochs_autoencoder,
        device=device,
        verbose=True,
    )
else:
    load_model_checkpoint(autoencoder, optimizer, os.path.join(os.dirname(__file__), "..", "..", "checkpoints", "autoencoder.pth.tar"))

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
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# Train denoising model
if args.train_denoiser:
    train_denoiser(
        model=denoise_model,
        autoencoder=autoencoder,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch_number=args.epochs_denoise,
        diffusion_timesteps=args.timesteps,
        beta_schedule=betas,
        device=device,
        verbose=True,
        checkpoint_path=os.path.join(os.dirname(__file__), "..", "..", "checkpoints", "denoise_model.pth.tar")
    )
else:
    load_model_checkpoint(denoise_model, optimizer, os.path.join(os.dirname(__file__), "..", "..", "checkpoints", "denoise_model.pth.tar"))

denoise_model.eval()

del train_loader, val_loader

# generate submission file
generate_submission(
    autoencoder=autoencoder,
    denoise_model=denoise_model,
    beta_schedule=betas,
    test_loader=test_loader,
    file_path=os.path.join(os.dirname(__file__), "..", "..", "submission.csv"),
    args=args,
    device=device,
)
