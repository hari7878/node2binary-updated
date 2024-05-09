import os
import click
import time
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt
from utils.graph_processing import process_edgelist
from utils.graph_processing import read_graph
from utils.random_walk import Walker
from utils.model_initialization_and_vocab import build_vocab
from utils.model_initialization_and_vocab import node2binaryParams
from utils.graph_processing import get_data
from utils.skipgram import SkipGrams
from utils.model_initialization_and_vocab import node2binary
from utils.model_train import Trainer
from utils.graph_processing import save_embeddings,load_embeddings,load_labels,load_embedding,split_train_test_graph
from utils.evaluation import NodeClassification,LinkPrediction
import random

# Main function!
@click.command(no_args_is_help=True)
@click.argument('input_file', type=click.Path(exists=True))
@click.option("--is_directed", type=bool, default=False)
@click.option("--is_weighted", type=bool, default=False)
@click.option("--input_labels_path",default=None)
@click.option("--embed_dim", required=True, type=int)
@click.option("--embed_max_norm", default=None)
@click.option("--input_node_ids_numbered", type=bool, default=True)
@click.option("--print_this_time", type=bool, default=True)
@click.option("--nodes_indexing_starting_from_1", type=bool, default=True)
## training parameters

@click.option("--pos_weight", required=True, type=int)
@click.option("--neg_weight", required=True, type=int)
@click.option("--gradient_quarter", required=True, type=int)
@click.option("--gradient_bias", type=int, default=2)
@click.option("--neg_samples", type=int, default=5)
@click.option("--min_freq", default=0, type=int)
@click.option("--ns-array-len", default=5_000_000, type=int)
@click.option("--tokenizer", default="basic_english", type=str)
@click.option("--batch_size", default=128, type=int)
@click.option("--n_epochs", required=True, type=int)
## walk parameters
@click.option("--walk_length", type=int, default=80)
@click.option("--number_walks", type=int, default=10)
@click.option("--window_size", type=int, default=10, required=True)#window size
@click.option("--p", type=float, default=1)
@click.option("--q", type=int, default=1)
## control of printing
@click.option("--verbose", type=int, default=1)
@click.option("--task", type=str, default=None)
@click.option("--testing_ratio", type=float, default=None)
@click.option('--binary_operator', type=click.Choice(['Average', 'Hadamard', 'Weighted-L1', 'Weighted-L2'], case_sensitive=False), default=None)
@click.option("--seed", type=int, default=42)





def main(input_file,is_directed,is_weighted,input_labels_path,embed_dim,embed_max_norm,input_node_ids_numbered,print_this_time,nodes_indexing_starting_from_1,pos_weight,neg_weight,gradient_quarter,gradient_bias,verbose,neg_samples,min_freq,ns_array_len,tokenizer,batch_size,n_epochs,walk_length,number_walks,window_size,p,q,task,testing_ratio,binary_operator,seed):
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_file_abs_path = os.path.abspath(input_file)
    base_name, ext = os.path.splitext(os.path.basename(input_file_abs_path))
    reverse_mapping_dict={}

    # Decide whether to process the file based on the input_node_ids_numbered flag
    if not input_node_ids_numbered:
        # If node IDs are not numbered, adjust the file and update the path to use the adjusted file
        final_input_edgelist_file = os.path.join(os.path.dirname(input_file_abs_path), f"{base_name}_adjusted.edgelist")
        print(final_input_edgelist_file)
        mapping_dict, reverse_mapping_dict = process_edgelist(input_file_abs_path, final_input_edgelist_file)
        # Use the adjusted file for further processing
        input_file_to_use = final_input_edgelist_file
    else:
        # If node IDs are numbered, use the original file as is
        input_file_to_use = input_file_abs_path
    if task=='link-prediction':
        G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph(os.path.abspath(input_file_to_use), seed,testing_ratio, is_weighted)
        input_file_to_use = train_graph_filename

    # Now, read the graph using the appropriate file
    nx_G = read_graph(input_file_to_use, is_directed, is_weighted)

    #Start Walks
    walker = Walker(nx_G, is_directed, p=p, q=q)
    print("Preprocess transition probs...")
    walker.preprocess_transition_probs()
    walks = walker.simulate_walks(num_walks=number_walks, walk_length=walk_length)
    print("\nWalks output:", np.array(walks).shape)

    #post-processing walks to start node indexing from 0
    if nodes_indexing_starting_from_1 or not input_node_ids_numbered:
        walks = [[element - 1 for element in row] for row in walks]

    # Get Model parameters
    params = node2binaryParams(
        pos_weight=pos_weight,
        neg_weight=neg_weight,
        gradient_quarter=gradient_quarter,
        gradient_bias=gradient_bias,
        verbose=verbose,
        print_this_time=print_this_time,
        MIN_FREQ=min_freq,
        SKIPGRAM_N_WORDS=window_size,
        NEG_SAMPLES=neg_samples,
        NS_ARRAY_LEN=ns_array_len,
        TOKENIZER=tokenizer,
        BATCH_SIZE=batch_size,
        EMBED_DIM=embed_dim,
        EMBED_MAX_NORM=embed_max_norm,
        N_EPOCHS=n_epochs,
        DEVICE=device
    )
    #Preprocess data obtained from walk as strings
    train_iter = get_data(walks)
    #Tokenize data
    tokenizer = get_tokenizer(params.TOKENIZER)
    #Map vocabulary to index and frequencies
    #vocab_words are list of nodes ordered by freq
    my_vocab, vocab_words = build_vocab(train_iter, tokenizer, params)
    #Compress contexts and subsampling
    skip_gram = SkipGrams(vocab=my_vocab, params=params, tokenizer=tokenizer)
    #Model initialization
    model = node2binary(vocab=my_vocab, params=params).to(params.DEVICE)
    print("Initialized.")
    print(time.asctime())
    print("trainer", end=" ")
    time.sleep(1)
    #Start model Training
    trainer = Trainer(
        model=model,
        params=params,
        train_iter=train_iter,
        valid_iter=train_iter,
        vocab=my_vocab,
        skipgrams=skip_gram
    )
    print("created")
    trainer.train()
    print("training complete")

    #Plot loss_train
    trainer.plot_loss()
    #Get embeddings
    final_embeddings = model.embeddings
    #Output_filename
    output_filename = f"embeddings/{base_name}_adjusted.txt"
    embedding_look_up=save_embeddings(vocab_words, final_embeddings, output_filename,reverse_mapping_dict,input_node_ids_numbered)
    #plot loss_train_validation
    plt.plot(trainer.loss['train'], label="Train")
    plt.plot(trainer.loss['valid'], label="Validation")
    plt.legend()
    plt.title('Loss plots')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plot_filename = os.path.join("loss_plots", "loss_plot_train_validation.png")
    plt.savefig(plot_filename)
    plt.show()

    #Downstream tasks
    if task== "node-classification":
        embedding_look_up = load_embeddings(os.path.abspath(output_filename))
        labels = load_labels(input_labels_path)
        node_list = list(embedding_look_up.keys())
        acc, micro, macro = NodeClassification(embedding_look_up, node_list, labels, testing_ratio, 0)
        print("############# NODE-CLASSIFICATION ###################")
        print(f"Accuracy: {acc}, Micro F1: {micro}, Macro F1: {macro}")

    if task == 'link-prediction':
        auc_roc, auc_pr, accuracy, f1 = LinkPrediction(embedding_look_up, G, G_train, testing_pos_edges, seed, binary_operator)
        print('#' * 9 + ' Link Prediction Performance ' + '#' * 9)
        print(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}')
        print('#' * 50)






main()
