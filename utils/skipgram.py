import torch
from .model_initialization_and_vocab import Vocab
import random
import numpy as np
from collections import Counter
from .model_initialization_and_vocab import node2binaryParams
class SkipGrams:  # it works like a custom dataloader
    def __init__(self, vocab: Vocab, params: node2binaryParams, tokenizer):
        self.vocab = vocab
        self.params = params
        self.tokenizer = tokenizer
        self.discard_probs = self._create_discard_dict()


    def _create_discard_dict(self):
        #Why do this?
        #if len(self.vocab) < 50:
            #return dict()
        discard_dict = {}
        #changes1
        for _, (word, freq) in self.vocab.itos.items():
            z_wi = freq / self.vocab.total_tokens
            p_wi = (pow(z_wi / 0.001, 0.5) + 1) * (0.001 / z_wi)
            discard_dict[word] = p_wi
        return discard_dict

    def collate_skipgram(self, batch):
        #Subsampling
        batch_input, batch_output = [], []
        for text in batch:
            #check this text tokens its based on rank indexing do we need it?
            text_tokens = [int(item) for item in text.split()]
            #text_tokens = self.vocab.get_index(self.tokenizer(text))


            if len(text_tokens) < self.params.SKIPGRAM_N_WORDS * 2 + 1:
                continue

            # creating contexts
            for idx in range(len(text_tokens) - self.params.SKIPGRAM_N_WORDS * 2):
                token_id_sequence = text_tokens[
                                    idx: (idx + self.params.SKIPGRAM_N_WORDS * 2 + 1)]
                input_ = token_id_sequence.pop(self.params.SKIPGRAM_N_WORDS)
                outputs = token_id_sequence


                prb = random.random()
                del_pair = self.discard_probs.get(str(input_))
                if del_pair <=prb:
                    continue
                else:
                    for output in outputs:
                        prb = random.random()
                        del_pair = self.discard_probs.get(str(output))
                        if del_pair <=prb:
                            continue
                        else:
                            batch_input.append(input_)
                            batch_output.append(output)


        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)

        return batch_input, batch_output

    def compress_contexts(self, batch_input_unsorted, batch_output_unsorted):
        # returns tuple( tuple((ctx_node, occurrences), ...) for node in batch_input )
        # print("Compressing", contexts)
        #Remove self contexts between contexts and target(both should not have same value)
        mask = batch_input_unsorted != batch_output_unsorted
        batch_input_unsorted = batch_input_unsorted[mask]
        batch_output_unsorted = batch_output_unsorted[mask]
        compressed = []
        temp_list = []
        #we need to sort both batch_input and batch_output together as one rather than sort each individually because we lose track of the index of context if we sort only target that why we are sorting based on argsort and sorting both at once
        sort_idx = np.argsort(batch_input_unsorted)

        batch_input = np.array(batch_input_unsorted)[sort_idx]
        batch_output = np.array(batch_output_unsorted)[sort_idx]


        for i in range(len(batch_input) - 1):

            if batch_input[i + 1] == batch_input[i]:
                temp_list.append(batch_output[i])
            else:
                temp_list.append(batch_output[i])

                #batch_input[i] is basically target value at i
                #context_dixt get frequency of each context for given target in dict form
                context_dict = Counter(sorted(temp_list))
                for key in context_dict.keys():
                    #[target,partiular_context,frequency_of_particular_context]
                    compressed.append([batch_input[i], key, context_dict[key]])
                temp_list = []
            if i == len(batch_input) - 2:
                temp_list.append(batch_output[i + 1])
                context_dict = Counter(sorted(temp_list))
                for key in context_dict.keys():
                    compressed.append([batch_input[i], key, context_dict[key]])

        compressed = torch.tensor(compressed, dtype=torch.int32, device=self.params.DEVICE)
        temp = compressed[compressed[:, 1].sort()[1]]
        sorted_compressed = temp[temp[:, 0].sort()[1]]

        return sorted_compressed
