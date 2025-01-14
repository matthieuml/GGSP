import os
import shutil
import networkx as nx
import random
import argparse

from tqdm import tqdm


def generate_augmentation_dataset(dataset_folder, dataset, multiplier_factor: int = 3):
    augmented_dataset = dataset + "_augmented"
    augmented_graph_path = os.path.join(dataset_folder, augmented_dataset, "graph")
    augmented_desc_path = os.path.join(dataset_folder, augmented_dataset, "description")

    # Create the directories for the augmented dataset
    os.makedirs(augmented_graph_path, exist_ok=True)
    os.makedirs(augmented_desc_path, exist_ok=True)

    graph_path = os.path.join(dataset_folder, dataset, "graph")
    desc_path = os.path.join(dataset_folder, dataset, "description")

    # Traverse through all the graphs in the folder
    files = [f for f in os.listdir(graph_path) if f.endswith(('.graphml', '.edgelist'))]

    new_graph_index = 0
    for fileread in tqdm(files, desc="Augmenting graphs"):
        tokens = fileread.split("/")
        idx = tokens[-1].find(".")
        filen = tokens[-1][:idx]
        extension = tokens[-1][idx + 1:]
        fread = os.path.join(graph_path, fileread)
        fstats = os.path.join(desc_path, filen + ".txt")

        # Load dataset to NetworkX
        if extension == "graphml":
            G = nx.read_graphml(fread)
        else:
            G = nx.read_edgelist(fread, data=True)  # Retain edge attributes if present

        # Generate a random permutation of the node indices
        for _ in range(multiplier_factor):
            mapping = {node: new_node for new_node, node in enumerate(random.sample(G.nodes(), len(G.nodes())))}
            G_aug = nx.relabel_nodes(G, mapping)

            # Save the augmented graph
            if extension == "graphml":
                nx.write_graphml(G_aug, os.path.join(augmented_graph_path, f"graph_{new_graph_index}.graphml"))
            else:
                nx.write_edgelist(G_aug, os.path.join(augmented_graph_path, f"graph_{new_graph_index}.graphml"), data=True)

            shutil.copy(fstats, os.path.join(augmented_desc_path, f"graph_{new_graph_index}.txt"))
            
            new_graph_index += 1

        # Copy original graph
        if extension == "graphml":
            nx.write_graphml(G, os.path.join(augmented_graph_path, f"graph_{new_graph_index}.graphml"))
        else:
            nx.write_edgelist(G, os.path.join(augmented_graph_path, f"graph_{new_graph_index}.graphml"), data=True)

        shutil.copy(fstats, os.path.join(augmented_desc_path, f"graph_{new_graph_index}.txt"))
      
        new_graph_index += 1
        
    print(f"Augmentation completed. Augmented dataset saved in: {os.path.join(dataset_folder, augmented_dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment graph dataset by permuting node indices.")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to augment")
    
    args = parser.parse_args()

    generate_augmentation_dataset(args.dataset_folder, args.dataset)