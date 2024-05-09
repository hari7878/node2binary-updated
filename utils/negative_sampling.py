from .model_initialization_and_vocab import Vocab
import random
import torch
import numpy as np
from collections import Counter
class NegativeSampler:
    def __init__(self, vocab: Vocab, ns_exponent: float, ns_array_len: int):
        self.vocab = vocab
        #self.params = params
        self.ns_exponent = ns_exponent
        self.ns_array_len = ns_array_len
        self.ns_array = self._create_negative_sampling()

    def __len__(self):
        return len(self.ns_array)

    def _create_negative_sampling(self):
        # Calculate the adjusted frequency for each word.#changes here done
        adjusted_freq = {int(word): freq ** self.ns_exponent for word, (_, freq) in self.vocab.stoi.items()}

        # Calculate the normalizing constant (denominator).
        normalizing_constant = sum(adjusted_freq.values())

        # Create the negative sampling distribution.
        ns_distribution = {word: (freq / normalizing_constant) for word, freq in adjusted_freq.items()}

        # Scale the distribution to the desired length of the negative sampling array.
        ns_array = []
        for word, prob in ns_distribution.items():
            # The expected count for this word in the sampling array.
            expected_count = int(round(prob * self.ns_array_len))
            ns_array.extend([word] * expected_count)

        # If the ns_array is longer than ns_array_len due to rounding, truncate it.
        ns_array = ns_array[:self.ns_array_len]
        #random.shuffle(ns_array)


        return ns_array

    def sample(self, inputs, contexts, n_batches: int = 1, n_samples: int = 1):
        samples = []
        for i in range(n_batches):
            # Start with a random sample
            sample = random.sample(self.ns_array, n_samples)
            # Check if the sample is the input node or one of the context nodes if so resample as we dont want positive and negative pair to be same
            while any(s == inputs[i] or s ==contexts[i] for s in sample):
                sample = random.sample(self.ns_array, n_samples)
            samples.append(sample)
        return torch.tensor(samples)

    def compress_neg_samples(self, batch_input_unsorted, neg_samples_unsorted):
        # returns tuple( tuple((ctx_node, occurrences), ...) for node in batch_input )
        # print("Compressing", contexts)
        #batch_input unsorted is your target words
        compressed = []
        temp_list = []
        #again we cant do normal sort based on only target words as we lose track of corresponding negative samples so by argsort we are simultaneously sorting both based on index
        sort_idx = np.argsort(batch_input_unsorted)

        batch_input = np.array(batch_input_unsorted)[sort_idx]
        neg_samples = np.array(neg_samples_unsorted)[sort_idx]

        # print("batch_input:", batch_input)
        # print("neg_samples:", neg_samples)

        for i in range(len(batch_input) - 1):

            if batch_input[i + 1] == batch_input[i]:
                temp_list.extend(neg_samples[i])
            else:
                temp_list.extend(neg_samples[i])
                neg_samples_dict = Counter(sorted(temp_list))
                for key in neg_samples_dict.keys():
                    compressed.append([batch_input[i], key, neg_samples_dict[key]])
                temp_list = []
            if i == len(batch_input) - 2:
                temp_list.extend(neg_samples[i + 1])
                neg_samples_dict = Counter(sorted(temp_list))
                for key in neg_samples_dict.keys():
                    compressed.append([batch_input[i], key, neg_samples_dict[key]])


        compressed = torch.tensor(compressed, dtype=torch.int32)
        temp = compressed[compressed[:, 1].sort()[1]]
        sorted_compressed = temp[temp[:, 0].sort()[1]]
        return sorted_compressed
