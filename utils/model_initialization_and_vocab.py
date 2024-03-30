from dataclasses import dataclass
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine
import numpy as np
from typing import Dict, List, Optional, Union
from collections import Counter,OrderedDict
import re
class node2binaryParams:
    def __init__(self, pos_weight=25, neg_weight=10, gradient_quarter=150000,
                 gradient_bias=2, verbose=2, print_this_time=True,
                 MIN_FREQ=0, SKIPGRAM_N_WORDS=10, NEG_SAMPLES=5,
                 NS_ARRAY_LEN=5_000_000, TOKENIZER='basic_english',
                 BATCH_SIZE=128, EMBED_DIM=128, EMBED_MAX_NORM=None,
                 N_EPOCHS=25, DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.gradient_quarter = gradient_quarter
        self.gradient_bias = gradient_bias
        self.verbose = verbose
        self.print_this_time = print_this_time
        self.MIN_FREQ = MIN_FREQ
        self.SKIPGRAM_N_WORDS = SKIPGRAM_N_WORDS
        self.NEG_SAMPLES = NEG_SAMPLES
        self.NS_ARRAY_LEN = NS_ARRAY_LEN
        self.TOKENIZER = TOKENIZER
        self.BATCH_SIZE = BATCH_SIZE
        self.EMBED_DIM = EMBED_DIM
        self.EMBED_MAX_NORM = EMBED_MAX_NORM
        self.N_EPOCHS = N_EPOCHS
        self.DEVICE = DEVICE
class Vocab:
    def __init__(self, list):
        self.stoi = {v[0]: (k, v[1]) for k, v in enumerate(list)}
        self.itos = {k: (v[0], v[1]) for k, v in enumerate(list)}

        print("Vocab:", list, "created")
        print("stoi:", self.stoi)
        print("itos:", self.itos)
        self.total_tokens = np.nansum(
            [f for _, (_, f) in self.stoi.items()]
            , dtype=int)

    def __len__(self):
        return len(self.stoi)

    def get_index(self, word: Union[str, List]):
        if isinstance(word, str):
            if word in self.stoi:
                return self.stoi.get(word)[0]
            else:
                raise ValueError(f"word {word} does not exist")
        elif isinstance(word, list):
            res = []
            for w in word:
                if w in self.stoi:
                    res.append(self.stoi.get(w)[0])
                else:
                    raise ValueError(f"word {w} does not exist")
            return res
        else:
            raise ValueError(
                f"Word {word} is not a string or a list of strings."
            )

    def get_freq(self, word: Union[str, List]):
        if isinstance(word, str):
            if word in self.stoi:
                return self.stoi.get(word)[1]
            else:
                raise ValueError(f"word {word} does not exist")
        elif isinstance(word, list):
            res = []
            for w in word:
                if w in self.stoi:
                    res.append(self.stoi.get(w)[1])
                else:
                    raise ValueError(f"word {w} does not exist")
            return res
        else:
            raise ValueError(
                f"Word {word} is not a string or a list of strings."
            )

    def lookup_token(self, token: Union[int, List]):
        if isinstance(token, (int, np.int64)):
            if token in self.itos:
                return self.itos.get(token)[0]
            else:
                raise ValueError(f"Token {token} not in vocabulary")
        elif isinstance(token, list):
            res = []
            for t in token:
                if t in self.itos:
                    res.append(self.itos.get(token)[0])
                else:
                    raise ValueError(f"Token {t} is not a valid index.")
            return res


def yield_tokens(iterator, tokenizer):
    r = re.compile('[a-z0-9]')
    for text in iterator:
        res = tokenizer(text)
        res = list(filter(r.match, res))

        yield res


def vocab(ordered_dict: Dict, min_freq: int = 1):
    tokens = []

    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append((token, freq))

    return Vocab(tokens)


def pipeline(word, vocab, tokenizer):
    return vocab(tokenizer(word))


def build_vocab(
        iterator,
        tokenizer,
        params: node2binaryParams,
        max_tokens: Optional[int] = None,
):
    counter = Counter()
    for tokens in yield_tokens(iterator, tokenizer):
        counter.update(tokens)

    # First sort by descending frequency, then lexicographically
    sorted_by_freq_tuples = sorted(
        counter.items(), key=lambda x: (-x[1], x[0])
    )

    ordered_dict = OrderedDict(sorted_by_freq_tuples)


    word_vocab = vocab(
        ordered_dict, min_freq=params.MIN_FREQ
    )

    return word_vocab, list(ordered_dict.keys())

class node2binary(nn.Module):
    def __init__(self, vocab: Vocab, params: node2binaryParams):
        super().__init__()
        self.vocab = vocab

        # Initialize random embeddings.
        self.embeddings = torch.randint(2,
                                        (self.vocab.__len__(), params.EMBED_DIM),
                                        dtype=torch.int32,
                                        device=params.DEVICE
                                        )

        '''
        self.c_embeddings = nn.Embedding(
            self.vocab.__len__()+1, 
            params.EMBED_DIM, 
            max_norm=params.EMBED_MAX_NORM
            )
        '''

    def embeddings(self):
        # embeddings = list(self.embeddings)
        embeddings = self.embeddings.cpu().detach().numpy()
        return embeddings

    def get_similar_words(self, word, n):
        word_id = self.vocab.get_index(word)

        embedding_norms = self.normalize_embeddings()
        word_vec = embedding_norms[word_id]
        word_vec = np.reshape(word_vec, (word_vec.shape[0], 1))
        dists = np.matmul(embedding_norms, word_vec).flatten()
        topN_ids = np.argsort(-dists)[1: n + 1]

        topN_dict = {}
        for sim_word_id in topN_ids:
            sim_word = self.vocab.lookup_token(sim_word_id)
            topN_dict[sim_word] = dists[sim_word_id]



        return topN_dict

    def get_similarity(self, word1, word2):
        idx1 = self.vocab.get_index(word1)
        idx2 = self.vocab.get_index(word2)

        embedding_norms = self.normalize_embeddings()
        word1_vec, word2_vec = embedding_norms[idx1], embedding_norms[idx2]

        return cosine(word1_vec, word2_vec)

class Vocab:
    def __init__(self, list):
        self.stoi = {v[0]: (k, v[1]) for k, v in enumerate(list)}
        self.itos = {k: (v[0], v[1]) for k, v in enumerate(list)}

        print("Vocab:", list, "created")
        print("stoi:", self.stoi)
        print("itos:", self.itos)
        self.total_tokens = np.nansum(
            [f for _, (_, f) in self.stoi.items()]
            , dtype=int)

    def __len__(self):
        return len(self.stoi)

    def get_index(self, word: Union[str, List]):
        if isinstance(word, str):
            if word in self.stoi:
                return self.stoi.get(word)[0]
            else:
                raise ValueError(f"word {word} does not exist")
        elif isinstance(word, list):
            res = []
            for w in word:
                if w in self.stoi:
                    res.append(self.stoi.get(w)[0])
                else:
                    raise ValueError(f"word {w} does not exist")
            return res
        else:
            raise ValueError(
                f"Word {word} is not a string or a list of strings."
            )

    def get_freq(self, word: Union[str, List]):
        if isinstance(word, str):
            if word in self.stoi:
                return self.stoi.get(word)[1]
            else:
                raise ValueError(f"word {word} does not exist")
        elif isinstance(word, list):
            res = []
            for w in word:
                if w in self.stoi:
                    res.append(self.stoi.get(w)[1])
                else:
                    raise ValueError(f"word {w} does not exist")
            return res
        else:
            raise ValueError(
                f"Word {word} is not a string or a list of strings."
            )

    def lookup_token(self, token: Union[int, List]):
        if isinstance(token, (int, np.int64)):
            if token in self.itos:
                return self.itos.get(token)[0]
            else:
                raise ValueError(f"Token {token} not in vocabulary")
        elif isinstance(token, list):
            res = []
            for t in token:
                if t in self.itos:
                    res.append(self.itos.get(token)[0])
                else:
                    raise ValueError(f"Token {t} is not a valid index.")
            return res


def yield_tokens(iterator, tokenizer):
    r = re.compile('[a-z0-9]')
    for text in iterator:
        res = tokenizer(text)
        res = list(filter(r.match, res))
        yield res


def vocab(ordered_dict: Dict, min_freq: int = 1):
    tokens = []

    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append((token, freq))

    return Vocab(tokens)


def pipeline(word, vocab, tokenizer):
    return vocab(tokenizer(word))


def build_vocab(
        iterator,
        tokenizer,
        params: node2binaryParams,
        max_tokens: Optional[int] = None,
):
    counter = Counter()
    for tokens in yield_tokens(iterator, tokenizer):
        counter.update(tokens)

    # First sort by descending frequency, then lexicographically
    sorted_by_freq_tuples = sorted(
        counter.items(), key=lambda x: (-x[1], x[0])
    )
    # print("sorted_by_freq_tuples:", sorted_by_freq_tuples)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)


    word_vocab = vocab(
        ordered_dict, min_freq=params.MIN_FREQ
    )

    return word_vocab, list(ordered_dict.keys())







