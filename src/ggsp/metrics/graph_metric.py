import networkx as nx
import numpy as np
import torch

from ggsp.utils import construct_nx_from_adj, compute_graph_features



def graph_norm(graphs1: nx.Graph, graphs2: nx.Graph, norm_type: str="L1"):
    """
    Evaluate the norm between the global features of two batches of graphs.
    Args:
        graphs1 (list of networkx.Graph): The first batch of graphs.
        graphs2 (list of networkx.Graph): The second batch of graphs.
        norm_type (str): The type of norm to compute ('L1', 'L2', 'MSE').
    Returns:
        torch.Tensor: A tensor of norms for each pair of graphs.
    """
    if len(graphs1) != len(graphs2):
        raise ValueError("Both batches must have the same number of graphs.")

    features1 = torch.stack([compute_graph_features(graph) for graph in graphs1])
    features2 = torch.stack([compute_graph_features(graph) for graph in graphs2])
    diff = features1 - features2

    if norm_type == "L1":
        norms = torch.sum(torch.abs(diff), dim=1)
    elif norm_type == "L2":
        norms = torch.sqrt(torch.sum(diff**2, dim=1))
    elif norm_type == "MSE":
        norms = torch.mean(diff**2, dim=1)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    return norms


def graph_norm_from_adj(adjs1:np.ndarray, adjs2:np.ndarray, norm_type:str="L1"):
    """
    Evaluate the norm between the global features of two batches of adj matrices.
    Args:
        graphs1 (list of networkx.Graph): The first batch of graphs.
        graphs2 (list of networkx.Graph): The second batch of graphs.
        norm_type (str): The type of norm to compute ('L1', 'L2', 'MSE').
    Returns:
        torch.Tensor: A tensor of norms for each pair of graphs.
    """
    if adjs1.shape != adjs2.shape:
        raise ValueError("Both batches must have the same number of graphs.")

    graphs1 = [construct_nx_from_adj(adj) for adj in adjs1]
    graphs2 = [construct_nx_from_adj(adj) for adj in adjs2]

    return graph_norm(graphs1, graphs2, norm_type)