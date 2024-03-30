import networkx as nx
import numpy as np
from torchtext.data import to_map_style_dataset
def process_edgelist(input_filename, output_filename):
    # Initialize dictionaries for mapping and reverse mapping
    mapping_dict = {}
    reverse_mapping_dict = {}
    current_id = 1

    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) != 3:  # Ensure each line has exactly 3 components
                continue

            # Extract nodes and weight
            node1, node2, weight = parts

            # Map node1
            if node1 not in mapping_dict:
                mapping_dict[node1] = current_id
                reverse_mapping_dict[current_id] = node1
                node1_mapped = current_id
                current_id += 1
            else:
                node1_mapped = mapping_dict[node1]

            # Map node2
            if node2 not in mapping_dict:
                mapping_dict[node2] = current_id
                reverse_mapping_dict[current_id] = node2
                node2_mapped = current_id
                current_id += 1
            else:
                node2_mapped = mapping_dict[node2]

            # Write the modified line to the new file
            outfile.write(f"{node1_mapped} {node2_mapped} {weight}\n")

    return mapping_dict, reverse_mapping_dict

def read_graph(input_file, is_directed, is_weighted):
    '''
    Reads the input network in networkx.
    '''

    if is_weighted:
        G = nx.read_edgelist(input_file, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not is_directed:
        G = G.to_undirected()

    return G

def get_data(walks):
    # gets the data
    walks_str = []
    for walk in walks:
        walks_str.append(' '.join(map(str, walk)))
    walks_str = np.array(walks_str)
    train_iter = walks_str
    train_iter = to_map_style_dataset(train_iter)

    return train_iter

import torch

def save_embeddings(vocab, final_embeddings, filename, reverse_mapping_dict=None, input_node_ids_numbered=True):
    # Ensure reverse_mapping_dict is a dictionary
    if not isinstance(reverse_mapping_dict, dict):
        raise ValueError("reverse_mapping_dict must be a dictionary")

    with open(filename, 'w') as fout:
        word_vectors = {}
        for i in range(len(final_embeddings)):
            # Convert PyTorch tensor to numpy array if necessary, then to list
            if isinstance(final_embeddings[i], torch.Tensor):
                # Convert tensor to list of 0s and 1s (assuming it's already in that format)
                vec = final_embeddings[i].tolist()
            else:
                # Handle non-tensor embeddings (if any) that are already lists
                vec = final_embeddings[i]
            word_vectors[vocab[i]] = vec
        node_num = len(word_vectors.keys())
        size = len(next(iter(word_vectors.values())))  # Getting the size from the first value
        fout.write(f"{node_num} {size}\n")
        for node, vec in word_vectors.items():
            if not input_node_ids_numbered:
                # Attempt to map back to original node ID using reverse_mapping_dict
                node_id = reverse_mapping_dict.get(node, node)  # Fallback to node if not found
            else:
                node_id = node
            fout.write(f"{node_id} {' '.join(map(str, vec))}\n")
