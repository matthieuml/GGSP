import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
import torch
from ggsp.utils import construct_nx_from_adj


def compute_features(graph):
    if isinstance(graph, torch.Tensor):
        graph = graph.detach().cpu().numpy()

    if isinstance(graph, np.ndarray) or isinstance(graph, torch.Tensor):
        graph = construct_nx_from_adj(graph)

    return np.array([graph.number_of_nodes(),
                     graph.number_of_edges(),
                     np.mean([deg for _, deg in graph.degree()]),
                     sum(nx.triangles(graph).values()) // 3,
                     nx.average_clustering(graph),
                     max(nx.core_number(graph).values()),
                     len(list(greedy_modularity_communities(graph)))])


def absolute_loss_features(graph_preds, graphs, data):
    """
    Compute the mean absolute error between two graphs
    """
    assert len(graph_preds) == len(graphs), "The number of graphs should be the same"

    if not isinstance(graph_preds, list) and len(graph_preds.shape) == 2:
        graph_preds = [graph_preds]
    if not isinstance(graphs, list) and len(graphs.shape) == 2:
        graphs = [graphs]

    N = len(graph_preds)

    features_1 = np.zeros((N, 7))
    features_2 = np.zeros((N, 7))

    for i in range(N):
        features_1[i] = compute_features(graph_preds[i])
        features_2[i] = compute_features(graphs[i])

    return abs(features_1 - features_2).sum(axis=1)


def mae(graph_preds, graphs):
    """
    Compute the mean absolute error between two graphs
    """
    return np.mean(absolute_loss_features(graph_preds, graphs))
