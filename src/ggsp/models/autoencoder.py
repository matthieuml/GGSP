import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool

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

    def loss_function(self, data, beta=0.05):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)

        recon = F.l1_loss(adj, data.A, reduction="mean")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta * kld

        return loss, recon, kld
