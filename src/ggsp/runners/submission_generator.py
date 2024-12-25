import csv
import torch
import argparse
from tqdm import tqdm
from typing import Union
from torch.utils.data import DataLoader

from ggsp.models import sample
from ggsp.utils import construct_nx_from_adj


def generate_submission(
    autoencoder: torch.nn.Module,
    denoise_model: torch.nn.Module,
    beta_schedule: torch.Tensor,
    test_loader: DataLoader,
    file_path: str,
    args: argparse.Namespace,
    device: Union[str, torch.device] = "cpu",
):
    """Generate submission file that should be uploaded to Kaggle.
    It consists of a CSV file with two columns: graph_id and edge_list.

    Args:
        autoencoder (torch.nn.Module): autoencoder model to decode the denoised latent vector
        denoise_model (torch.nn.Module): denoiser model to denoise the noisy data
        beta_schedule (torch.Tensor): noising beta schedule
        test_loader (DataLoader): test dataloader
        file_path (str): path to save the submission file
        args (argparse.Namespace): arguments
        device (Union[str, torch.device], optional): device. Defaults to "cpu".
    """

    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["graph_id", "edge_list"])
        for k, data in enumerate(
            tqdm(
                test_loader,
                desc="Processing test set",
            )
        ):
            data = data.to(device)

            stat = data.stats
            bs = stat.size(0)

            graph_ids = data.filename

            samples = sample(
                denoise_model,
                data.stats,
                latent_dim=args.latent_dim,
                timesteps=args.timesteps,
                betas=beta_schedule,
                batch_size=bs,
            )
            x_sample = samples[-1]
            adj = autoencoder.decode_mu(x_sample)
            stat_d = torch.reshape(stat, (-1, args.n_condition))

            for i in range(stat.size(0)):
                stat_x = stat_d[i]

                Gs_generated = construct_nx_from_adj(
                    adj[i, :, :].detach().cpu().numpy()
                )
                stat_x = stat_x.detach().cpu().numpy()

                # Define a graph ID
                graph_id = graph_ids[i]

                # Convert the edge list to a single string
                edge_list_text = ", ".join(
                    [f"({u}, {v})" for u, v in Gs_generated.edges()]
                )
                # Write the graph ID and the full edge list as a single row
                writer.writerow([graph_id, edge_list_text])
