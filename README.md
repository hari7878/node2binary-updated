# node2binary
Representing each node in a graph as d-dimensional binary vectors.<br />
## Parameters Example use case for yeast ppi dataset
python main.py C:\Users\haris\Downloads\node2binary-updated-main\node2binary-updated-main\data\HT.Combined.PMID-14690591.144gene.688link_adjusted.edgelist --is_weighted True  --embed_dim 128 --input_node_ids_numbered False --nodes_indexing_starting_from_1 False --pos_weight 25 --neg_weight 5 --gradient_quarter 150000 --gradient_bias 2 --neg_samples 5 --n_epochs 25 --p 4 --q 1  

## Data Parameters
'input_file': input data file path. <br /> 
"--is_weighted": Is the graph dataset weighted or not (True or False).<br />
"--input_labels_path": For multilabel node classification pass the label file path(default=None).<br />
"--embed_dim": Number of dimensions to embed the binary vector representations.<br />
"--input_node_ids_numbered": Are the nodes numbers in a dataset (True or False).<br />
“--nodes_indexing_starting_from_1”:  Are the nodes indexed starting from 1 (True or False).<br />

## Training Parameters
"--pos_weight" :Weight of the in-context nodes gradient matrix.<br />
"--neg_weight":Weight of the out-of context(negative samples) nodes gradient matrix.<br />
“--gradient_quarter”: Behaves similar to the learning rate.<br />
“—gradient_bias”: learning bias allows the algorithm to flip bits with zero gradient (to avoid local optima).<br />
"--neg_samples": Number of negative samples for each in context-node.<br />
"--batch_size": Size of each input training batch.<br />
"--n_epochs": number of epochs to the node2binary algorithm.<br />

## Walk Parameters
"--walk_length" : Length of each random walk from a reference node.(same as node2vec).<br />
"--number_walks": Number of random walks over each reference node (same as node2vec).<br />
"--window_size": Window length to define the in context nodes (same as node2vec).<br />
--p: return parameter in the random walk (same as node2vec).<br />
--q : walk away parameter in the random walk(same as node2vec).<br />

## Downstream tasks parameters
--task : Which downstream task to perform node-classification or link-prediction (default=None)<br />
--testing_ratio: Percentage of edges used as the test set for link-prediction<br />
'--binary_operator': Transformation to be applied to the node embeddings that is passed as input to logistic regression, choices= ['Average', 'Hadamard', 'Weighted-L1', 'Weighted-L2']<br />
