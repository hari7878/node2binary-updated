import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import random
from .graph_processing import load_embeddings
from .graph_processing import load_labels
from .graph_processing import get_y_pred
from .graph_processing import generate_neg_edges

def NodeClassification(embedding_look_up, node_list, labels, testing_ratio, seed):
    n_splits = 10
    ss = ShuffleSplit(n_splits=n_splits, test_size=testing_ratio, random_state=seed)

    # Initialize scores
    accuracy_scores = []
    micro_f1_scores = []
    macro_f1_scores = []

    # Initialize binarizer
    binarizer = MultiLabelBinarizer(sparse_output=True)
    all_labels = [labels[node] for node in node_list]
    binarizer.fit(all_labels)

    # Perform the splits and training/testing
    for train_index, test_index in ss.split(node_list):
        X_train = [embedding_look_up[node_list[i]] for i in train_index]
        y_train = binarizer.transform([labels[node_list[i]] for i in train_index]).todense()
        y_train = np.asarray(y_train)  # Convert to np.array
        X_test = [embedding_look_up[node_list[i]] for i in test_index]
        y_test = binarizer.transform([labels[node_list[i]] for i in test_index]).todense()
        y_test = np.asarray(y_test)
        model = OneVsRestClassifier(LogisticRegression(random_state=seed, solver='lbfgs'))
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)

        # Use a small trick: assume we know how many labels to predict
        y_pred = get_y_pred(y_test, y_pred_prob)

        # Store the scores
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        micro_f1_scores.append(f1_score(y_test, y_pred, average="micro"))
        macro_f1_scores.append(f1_score(y_test, y_pred, average="macro"))

    # Print and return the average performance
    avg_accuracy = np.mean(accuracy_scores)
    avg_micro_f1 = np.mean(micro_f1_scores)
    avg_macro_f1 = np.mean(macro_f1_scores)

    return avg_accuracy, avg_micro_f1, avg_macro_f1


def LinkPrediction(embedding_look_up, original_graph, train_graph, test_pos_edges, seed,binary_operator):
    random.seed(seed)
    import copy
    train_neg_edges = generate_neg_edges(original_graph, len(train_graph.edges()), seed)

    # create a auxiliary graph to ensure that testing negative edges will not used in training
    G_aux = copy.deepcopy(original_graph)
    G_aux.add_edges_from(train_neg_edges)
    test_neg_edges = generate_neg_edges(G_aux, len(test_pos_edges), seed)

    # construct X_train, y_train, X_test, y_test
    X_train = []
    y_train = []
    for edge in train_graph.edges():
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = binary_operation(np.array(node_u_emb), np.array(node_v_emb), binary_operator)
        X_train.append(feature_vector)
        y_train.append(1)
    for edge in train_neg_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = binary_operation(np.array(node_u_emb), np.array(node_v_emb), binary_operator)
        X_train.append(feature_vector)
        y_train.append(0)

    X_test = []
    y_test = []
    for edge in test_pos_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = binary_operation(np.array(node_u_emb), np.array(node_v_emb), binary_operator)
        X_test.append(feature_vector)
        y_test.append(1)
    for edge in test_neg_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = binary_operation(np.array(node_u_emb), np.array(node_v_emb), binary_operator)
        X_test.append(feature_vector)
        y_test.append(0)

    # shuffle for training and testing
    c = list(zip(X_train, y_train))
    random.shuffle(c)
    X_train, y_train = zip(*c)

    c = list(zip(X_test, y_test))
    random.shuffle(c)
    X_test, y_test = zip(*c)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    clf1 = LogisticRegression(random_state=seed, solver='lbfgs')
    clf1.fit(X_train, y_train)
    y_pred_proba = clf1.predict_proba(X_test)[:, 1]
    y_pred = clf1.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return auc_roc, auc_pr, accuracy, f1

def binary_operation(u_emb, v_emb, operator):
    if operator == 'Average':
        return (u_emb + v_emb) / 2
    elif operator == 'Hadamard':
        return np.multiply(u_emb, v_emb)
    elif operator == 'Weighted-L1':
        return np.abs(u_emb - v_emb)
    elif operator == 'Weighted-L2':
        return (u_emb - v_emb) ** 2
    else:  # Default action if none of the conditions are met
        # Assuming default action is concatenation
        return np.append(u_emb, v_emb)
