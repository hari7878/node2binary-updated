import networkx as nx
import numpy as np
from torchtext.data import to_map_style_dataset
import random
import itertools
import copy
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
            outfile.write(f"{node1_mapped} {node2_mapped} {float(weight)}\n")

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
    if not isinstance(reverse_mapping_dict, dict) and not input_node_ids_numbered:
        raise ValueError("reverse_mapping_dict must be provided and be a dictionary when input_node_ids_numbered is False")

    # Initialize the embedding lookup dictionary
    embedding_lookup = {}

    with open(filename, 'w') as fout:
        # Calculate the number of nodes and the embedding size
        node_num = len(final_embeddings)
        size = len(final_embeddings[0]) if node_num > 0 else 0
        fout.write(f"{node_num} {size}\n")

        for i in range(node_num):
            node_id = i + 1  # Node IDs start from 1
            if not input_node_ids_numbered:
                # Use reverse mapping to find the original node ID
                original_node_id = reverse_mapping_dict.get(node_id, node_id)
                node_id_str = str(original_node_id)  # Ensure node_id is a string for the dictionary
            else:
                node_id_str = str(node_id)

            # Convert PyTorch tensor to list if necessary
            vec = final_embeddings[i].tolist() if isinstance(final_embeddings[i], torch.Tensor) else final_embeddings[i]

            # Write the embedding to the file
            fout.write(f"{node_id_str} {' '.join(map(str, vec))}\n")

            # Update the embedding lookup dictionary
            embedding_lookup[str(node_id)] = vec

    return embedding_lookup

def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line
        for line in file:
            parts = line.strip().split()
            node_id = parts[0]  # Node ID as string
            vector = [float(x) for x in parts[1:]]  # Convert the rest to float
            embeddings[node_id] = vector
    return embeddings

def load_labels(file_path):
    labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            node_id = parts[0]
            node_labels = [int(label) for label in parts[1:]]
            labels[node_id] = node_labels
    return labels

def get_y_pred(y_test, y_pred_prob):
    y_pred = np.zeros(y_pred_prob.shape)
    sort_index = np.flip(np.argsort(y_pred_prob, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = np.sum(y_test[i])
        for j in range(num):
            y_pred[i][sort_index[i][j]] = 1
    return y_pred

def generate_neg_edges(original_graph, testing_edges_num, seed):
    L = list(original_graph.nodes())

    # create a complete graph
    G = nx.Graph()
    G.add_nodes_from(L)
    G.add_edges_from(itertools.combinations(L, 2))
    # remove original edges
    G.remove_edges_from(original_graph.edges())
    random.seed(seed)
    neg_edges = random.sample(G.edges, testing_edges_num)
    return neg_edges

def load_embedding(embedding_file_name, node_list=None):
    with open(embedding_file_name) as f:
        node_num, emb_size = f.readline().split()
        print('Nodes with embedding: %s'%node_num)
        embedding_look_up = {}
        if node_list:
            for line in f:
                vec = line.strip().split()
                node_id = vec[0]
                if (node_id in node_list):
                    emb = [float(x) for x in vec[1:]]
                    emb = emb / np.linalg.norm(emb)
                    emb[np.isnan(emb)] = 0
                    embedding_look_up[node_id] = np.array(emb)

def split_train_test_graph(input_edgelist, seed, testing_ratio, weighted):
    if (weighted):
        G = nx.read_weighted_edgelist(input_edgelist)
    else:
        G = nx.read_edgelist(input_edgelist)
    node_num1, edge_num1 = len(G.nodes), len(G.edges)
    print('Original Graph: nodes:', node_num1, 'edges:', edge_num1)
    testing_edges_num = int(len(G.edges) * testing_ratio)
    random.seed(seed)
    testing_pos_edges = random.sample(G.edges, testing_edges_num)
    G_train = copy.deepcopy(G)
    for edge in testing_pos_edges:
        node_u, node_v = edge
        if (G_train.degree(node_u) > 1 and G_train.degree(node_v) > 1):
            G_train.remove_edge(node_u, node_v)
            if G_train.degree(node_u) == 0 or G_train.degree(node_v) == 0:
                G_train.add_edge(node_u, node_v, weight=1)

    G_train.remove_nodes_from(nx.isolates(G_train))
    node_num2, edge_num2 = len(G_train.nodes), len(G_train.edges)
    assert node_num1 == node_num2
    train_graph_filename = 'graph_train.edgelist'
    if weighted:
        nx.write_edgelist(G_train, train_graph_filename, data=['weight'])
    else:
        nx.write_edgelist(G_train, train_graph_filename, data=False)

    node_num1, edge_num1 = len(G_train.nodes), len(G_train.edges)
    print('Training Graph: nodes:', node_num1, 'edges:', edge_num1)
    return G, G_train, testing_pos_edges, train_graph_filename
