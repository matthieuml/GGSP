import torch
import torch.nn as nn
import torch.nn.functional as F

from ggsp.models.decoders import *
from ggsp.models.encoders import *

# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim_enc,
        hidden_dim_dec,
        latent_dim,
        n_layers_enc,
        n_layers_dec,
        n_max_nodes,
        encoder_class_name="GIN",
        decoder_class_name="Decoder",
        kld_weight=0.05,
    ):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = globals()[encoder_class_name](
            input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc
        )
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = globals()[decoder_class_name](
            latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes
        )
        self.beta = kld_weight

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.0):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def decode_mu(self, mu):
        adj = self.decoder(mu)
        return adj
    
    def beta_step(self):
        self.beta *= (1 + 0.075)
        print(f"New beta: {self.beta}") 

    def loss_function(self, data, k=2):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)

        recon = F.l1_loss(adj, data.A, reduction="mean")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon + self.beta * kld

        # contrastive loss
        if k >= 0:
            temperature = 0.07
            # Find the nearest neighbors
            n = data.stats.shape[0]
            distance = ((data.stats[None, :] - data.stats[:, None])**2).sum(axis=-1)
            _, neighbors_indices = torch.topk(distance, k, largest=False)

            # Create the mask
            mask = torch.zeros((n, n), dtype=torch.bool)
            row_indices = torch.arange(n).unsqueeze(1).expand_as(neighbors_indices)
            mask[row_indices, neighbors_indices] = True
            mask.fill_diagonal_(False)

            # Compute the cosine similarity
            x_g_normalized = F.normalize(x_g, dim=1)
            cosine_similarity = x_g_normalized @ x_g_normalized.T

            # Compute the logits
            logits = torch.exp(cosine_similarity / temperature)
            numerator = logits 
            denominator = (logits * (1-torch.eye(n))).sum(dim=-1, keepdim=True)

            # Compute the contrastive loss
            contrastive_loss = -(mask * torch.log(numerator / denominator)).mean()
            loss += contrastive_loss
        
        else:
            contrastive_loss = torch.Tensor([0])

        return loss, recon, kld, contrastive_loss
