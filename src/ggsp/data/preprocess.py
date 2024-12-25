import os
import networkx as nx
import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F

from tqdm import tqdm
import scipy.sparse as sparse
from torch_geometric.data import Data

from ggsp.data import extract_features_from_file, extract_numbers_from_text


def preprocess_dataset(dataset_folder, dataset, n_max_nodes, spectral_emb_dim):
    data_lst = []
    filename = os.path.join(dataset_folder, "dataset_" + dataset + ".pt")
    if dataset == "test":
        desc_file = "./data/" + dataset + "/test.txt"
        desc_file = os.path.join(dataset_folder, dataset, "test.txt")

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f"Dataset {filename} loaded from file")

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)
                feats_stats = extract_numbers_from_text(desc)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                data_lst.append(Data(stats=feats_stats, filename=graph_id))
            fr.close()
            torch.save(data_lst, filename)
            print(f"Dataset {filename} saved")

    else:
        graph_path = os.path.join(dataset_folder, dataset, "graph")
        desc_path = os.path.join(dataset_folder, dataset, "description")

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f"Dataset {filename} loaded from file")

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path)]
            adjs = []
            eigvals = []
            eigvecs = []
            n_nodes = []
            max_eigval = 0
            min_eigval = 0
            for fileread in tqdm(files, desc="Building graphs from files"):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx + 1 :]
                fread = os.path.join(graph_path, fileread)
                fstats = os.path.join(desc_path, filen + ".txt")
                # load dataset to networkx
                if extension == "graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                        node_degree_list, key=lambda tt: tt[1], reverse=True
                    )

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

                adj = torch.from_numpy(adj_bfs).float()
                diags = np.sum(adj_bfs, axis=0)
                diags = np.squeeze(np.asarray(diags))
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with sp.errstate(divide="ignore"):
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()
                eigval, eigvecs = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs = torch.real(eigvecs)
                idx = torch.argsort(eigval)
                eigvecs = eigvecs[:, idx]

                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim + 1)
                x[:, 0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:, 0] / (
                    n_max_nodes - 1
                )
                mn = min(G.number_of_nodes(), spectral_emb_dim)
                mn += 1
                x[:, 1:mn] = eigvecs[:, :spectral_emb_dim]
                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_features_from_file(fstats)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                data_lst.append(
                    Data(
                        x=x,
                        edge_index=edge_index,
                        A=adj,
                        stats=feats_stats,
                        filename=filen,
                    )
                )
            torch.save(data_lst, filename)
            print(f"Dataset {filename} saved")
    return data_lst
