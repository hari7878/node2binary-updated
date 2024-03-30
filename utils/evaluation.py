import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

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