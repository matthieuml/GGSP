import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout

        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [
            nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers - 2)
        ]
        mlp_layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.relu(self.mlp[i](x))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.mlp[self.n_layers - 1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:, :, 0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj

class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        """
        GNN-based Decoder for reconstructing the adjacency matrix from a latent representation.

        Args:
            latent_dim (int): Dimension of the latent space.
            hidden_dim (int): Dimension of the hidden layers.
            n_layers (int): Number of GNN layers.
            n_nodes (int): Number of nodes in the graph.
        """
        super(GNNDecoder, self).__init__()

        # Linear layer to initialize node embeddings from latent space
        self.node_embedding = nn.Linear(latent_dim, hidden_dim)

        # Define a list of GNN layers (e.g., GCNConv)
        self.gnn_layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)])

        # MLP to predict edges between node pairs
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # Combine features from two nodes
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),              # Output a single value for edge probability
            nn.Sigmoid()                           # Output in the range [0, 1]
        )

        self.n_nodes = n_nodes  # Number of nodes in the graph

    def forward(self, z):
        """
        Forward pass to decode the latent vector into an adjacency matrix.

        Args:
            z (torch.Tensor): Latent representation of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed adjacency matrix of shape (batch_size, n_nodes, n_nodes).
        """
        # Expand latent vector into initial node features
        node_features = self.node_embedding(z).repeat(self.n_nodes, 1, 1).permute(1, 0, 2)

        # Get fully connected edge indices for the graph
        edge_index = self._get_fully_connected_edges(self.n_nodes, device=z.device)

        # Pass node features through the GNN layers
        for gnn in self.gnn_layers:
            # Flatten the batch for GNN input, then reshape back after GNN processing
            node_features = F.relu(gnn(node_features.reshape(-1, node_features.size(-1)), edge_index))
            node_features = node_features.reshape(z.size(0), self.n_nodes, -1)

        # Predict the adjacency matrix using the learned node features
        adj = self._predict_adjacency_matrix(node_features)
        return adj

    def _get_fully_connected_edges(self, n_nodes, device):
        """
        Generate a fully connected graph's edge index for message passing.

        Args:
            n_nodes (int): Number of nodes.
            device (torch.device): Device to store the tensor.

        Returns:
            torch.Tensor: Edge index of shape (2, num_edges) representing a fully connected graph.
        """
        return torch.combinations(torch.arange(n_nodes, device=device), r=2).t()

    def _predict_adjacency_matrix(self, node_features):
        """
        Predict the adjacency matrix from node features.

        Args:
            node_features (torch.Tensor): Node features of shape (batch_size, n_nodes, hidden_dim).

        Returns:
            torch.Tensor: Reconstructed adjacency matrix of shape (batch_size, n_nodes, n_nodes).
        """
        # Initialize an empty adjacency matrix
        adj = torch.zeros(node_features.size(0), self.n_nodes, self.n_nodes, device=node_features.device)

        # Iterate over all pairs of nodes (upper triangular part of the adjacency matrix)
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                # Concatenate the features of node pair (i, j)
                edge_feat = torch.cat([node_features[:, i, :], node_features[:, j, :]], dim=-1)

                # Predict edge existence (probability) using the edge predictor
                adj[:, i, j] = self.edge_predictor(edge_feat).squeeze()

                # Ensure symmetry for undirected graphs
                adj[:, j, i] = adj[:, i, j]

        return adj
