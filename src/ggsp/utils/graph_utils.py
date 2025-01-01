import networkx as nx
import torch
import numpy as np

def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G


def compute_graph_features(graph):
    """
    Compute the vector of global graph features.
    Args:
        graph (networkx.Graph): The input graph.
    Returns:
        torch.Tensor: A tensor of seven graph metrics.
    """
    try:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        avg_degree = np.mean([deg for _, deg in graph.degree()])
        num_triangles = sum(nx.triangles(graph).values()) // 3
        avg_clustering = nx.average_clustering(graph)
        max_core = max(nx.core_number(graph).values())
        num_communities = len(list(nx.algorithms.community.greedy_modularity_communities(graph)))

        return torch.tensor([
            num_nodes,
            num_edges,
            avg_degree,
            num_triangles,
            avg_clustering,
            max_core,
            num_communities
        ], dtype=torch.float32)
    except Exception as e:
        raise ValueError(f"Error computing graph features: {e}")
