from utils.model_initialization_and_vocab import node2binary,node2binaryParams
from utils.model_initialization_and_vocab import Vocab
from utils.skipgram import SkipGrams
from torch.utils.data import DataLoader
from time import monotonic
import torch
from utils.gradient import gradient
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.negative_sampling import NegativeSampler
class Trainer:
    def __init__(self, model: node2binary, params: node2binaryParams,
                 vocab: Vocab, train_iter, valid_iter, skipgrams: SkipGrams):
        self.model = model
        self.vocab = vocab
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.skipgrams = skipgrams
        self.params = params
        self.random_bound = 1 << 29

        self.epoch_train_mins = {}
        self.loss = {"train": [], "valid": []}

        # sending all to device
        self.model.to(self.params.DEVICE)

        self.negative_sampler = NegativeSampler(
            vocab=self.vocab, ns_exponent=.75,
            ns_array_len=self.params.NS_ARRAY_LEN
        )


    def train(self):

        for epoch in range(self.params.N_EPOCHS):
            # Generate Dataloaders
            self.train_dataloader = DataLoader(
                self.train_iter,
                batch_size=self.params.BATCH_SIZE,
                shuffle=False,
                collate_fn=self.skipgrams.collate_skipgram
            )
            '''
            self.valid_dataloader = DataLoader(
                self.valid_iter,
                batch_size=self.params.BATCH_SIZE,
                shuffle=False,
                collate_fn=self.skipgrams.collate_skipgram
            )
            '''
            # training the model
            st_time = monotonic()
            self._train_epoch()
            self.epoch_train_mins[epoch] = round((monotonic() - st_time) / 60, 1)

            # validating the model

            print(f"""Epoch: {epoch + 1}/{self.params.N_EPOCHS}\n""",
                  f"""    Train Loss: {self.loss['train'][-1]:.2}\n""",
                  # f"""    Valid Loss: {self.loss['valid'][-1]:.2}\n""",
                  f"""    Training Time (mins): {self.epoch_train_mins.get(epoch)}"""
                  """\n"""
                  )


    def calculate_loss(self, embeddings, positive_samples, negative_samples):

        positive_samples = torch.tensor(positive_samples, dtype=torch.int64)
        negative_samples = torch.tensor(negative_samples, dtype=torch.int64)

        # calculate loss
        # Get unique start nodes from both positive and negative samples
        start_nodes = positive_samples[:, 0].unique()


        pos_loss_avg = []
        neg_loss_avg = []

        # Calculate loss for each start node
        for v in start_nodes:
            pos_loss = 0.0
            neg_loss = 0.0

            # Get positive samples for this start node
            pos_mask = positive_samples[:, 0] == v
            # print("pos_mask:", pos_mask)
            pos_contexts = positive_samples[pos_mask][:, 1]
            # print("pos_contexts:", pos_contexts)
            pos_counts = positive_samples[pos_mask][:, 2].float()
            # print("pos_counts:", pos_counts)

            # Get negative samples for this start node
            neg_mask = negative_samples[:, 0] == v
            # print("neg_mask:", neg_mask)
            neg_contexts = negative_samples[neg_mask][:, 1]
            # print("neg_contexts:", neg_contexts)
            neg_counts = negative_samples[neg_mask][:, 2].float()
            # print("neg_counts:", neg_counts)

            # Get binary embeddings and calculate Hamming distances for positive samples

            if len(pos_contexts) > 0:
                v_emb_pos = embeddings[v].unsqueeze(0)
                u_emb_pos = embeddings[pos_contexts]
                hamming_dist_pos = (v_emb_pos != u_emb_pos).sum(dim=1).float()

                pos_loss += torch.sum((hamming_dist_pos ** 2) * pos_counts)

                pos_loss_avg.append((pos_loss / torch.sum(pos_counts)).item())

            # Calculate binary embeddings and calculate Hamming distances for negative samples
            if len(neg_contexts) > 0:
                v_emb_neg = embeddings[v].unsqueeze(0)
                u_emb_neg = embeddings[neg_contexts]
                hamming_dist_neg = (v_emb_neg != u_emb_neg).sum(dim=1).float()
                neg_loss += torch.sum((hamming_dist_neg ** 2) * neg_counts)
                neg_loss_avg.append((neg_loss / torch.sum(neg_counts)).item())

        pos_loss_avg = torch.tensor(pos_loss_avg, dtype=torch.float32)
        neg_loss_avg = torch.tensor(neg_loss_avg, dtype=torch.float32)

        loss = torch.subtract(torch.sum(pos_loss_avg), torch.sum(neg_loss_avg))

        return loss

    def train_one_batch(self, context_tensor, negative_samples,
                        pos_weight=1, neg_weight=1,  # both numbers should be positive
                        neg_sample_multiplier=2, gradient_quarter=100, bias=1, verbose=1):



        # get the gradients
        pos_gradient = gradient(self.model.embeddings, context_tensor, verbose=verbose)
        neg_gradient = gradient(self.model.embeddings, negative_samples, verbose=verbose)

        # our objective function is a loss function
        # so we want to subtract the positive sample gradient
        overall_gradient = neg_gradient * neg_weight - pos_gradient * pos_weight

        #if verbose >= 2:
            #print(overall_gradient)
            #print("p(flip) = g / (2*g + 2*{})".format(gradient_quarter))

        # add bias for zero, but keep negatives at exactly zero
        overall_gradient = torch.where(overall_gradient >= 0, overall_gradient + bias, 0)

        use_tanh = False
        if use_tanh:
            random_tensor = torch.rand(self.model.embeddings.size(), device=self.model.embeddings.device)
            probs = (1 / 2) * torch.tanh(2 * overall_gradient / gradient_quarter)
            flip_positions = (random_tensor < probs)
        else:
            # Hopefully this is accurate enough for small values of 2*g + 2*quarter
            random_tensor = torch.randint_like(self.model.embeddings, self.random_bound)
            flip_positions = (random_tensor % (2 * (overall_gradient + gradient_quarter))) < overall_gradient

        # flip the bits

        self.model.embeddings.bitwise_xor_(flip_positions.to(dtype=torch.int32))

        #if verbose >= 2:
            #print("Flipping positions ({} total flips):".format(torch.sum(flip_positions).item()))
            #print(flip_positions.to(dtype=torch.int32))
        #elif verbose >= 1:
            #print("Flipping {} bits".format(torch.sum(flip_positions, dtype=torch.int64).item()))

        # return the embeddings and some other helpful info
        '''
        (self.model.embeddings, {
            "neg_context": compressed_neg_samples,
            "flip_positions": flip_positions
        })
        '''

    def _train_epoch(self):
        running_loss = []
        # print(next(iter(self.train_dataloader)))
        for i, batch_data in enumerate(self.train_dataloader, 1):
            if len(batch_data[0]) == 0:
                continue
            #Repeated target vectors
            inputs = batch_data[0].cpu().detach().numpy()
            contexts = batch_data[1].cpu().detach().numpy()
            compressed_contexts = self.skipgrams.compress_contexts(inputs, contexts)
            compressed_contexts = compressed_contexts.to(self.params.DEVICE)
            # Generate new negative samples
            neg_samples = self.negative_sampler.sample(inputs,contexts,
                contexts.shape[0], self.params.NEG_SAMPLES)
            neg_samples = neg_samples.cpu().detach().numpy()
            compressed_neg_samples = self.negative_sampler.compress_neg_samples(inputs, neg_samples)
            compressed_neg_samples = compressed_neg_samples.to(self.params.DEVICE)

            # Calculate loss
            loss = self.calculate_loss(self.model.embeddings, compressed_contexts, compressed_neg_samples)
            running_loss.append(loss.item())

            # Training: gradient update
            t_start = time.time()
            loss = self.train_one_batch(compressed_contexts, compressed_neg_samples,
                                        pos_weight=self.params.pos_weight, neg_weight=self.params.neg_weight,
                                        neg_sample_multiplier=self.params.NEG_SAMPLES,
                                        gradient_quarter=self.params.gradient_quarter, bias=self.params.gradient_bias,
                                        verbose=self.params.verbose if self.params.print_this_time else 0)
            train_time = time.time() - t_start


        epoch_loss = np.mean(running_loss)
        self.loss['train'].append(epoch_loss)

    def _validate_epoch(self):
        running_loss = []

        print(self.valid_dataloader)

        for i, batch_data in enumerate(self.valid_dataloader, 1):
            if len(batch_data[0]) == 0:
                continue
            inputs = batch_data[0].cpu().detach().numpy()
            contexts = batch_data[1].cpu().detach().numpy()
            compressed_contexts = self.skipgrams.compress_contexts(inputs, contexts)
            compressed_contexts = compressed_contexts.to(self.params.DEVICE)
            # Generate new negative samples
            neg_samples = self.negative_sampler.sample(inputs,contexts,
                contexts.shape[0], self.params.NEG_SAMPLES)
            neg_samples = neg_samples.cpu().detach().numpy()
            compressed_neg_samples = self.negative_sampler.compress_neg_samples(inputs, neg_samples)
            compressed_neg_samples = compressed_neg_samples.to(self.params.DEVICE)

            # Calculate loss
            loss = self.calculate_loss(self.model.embeddings, compressed_contexts, compressed_neg_samples)
            running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss)
        self.loss['valid'].append(epoch_loss)

    # plot train and validation loss
    def plot_loss(self):

        plt.plot(self.loss['train'], label="Train")
        # plt.plot(self.loss['valid'], label="Validation")
        plt.legend()
        plt.title('Loss plot')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plot_filename = os.path.join("loss_plots", "loss_plot_train.png")
        plt.savefig(plot_filename)
        plt.show()

    '''
    def test_testwords(self, n: int = 5):
        for word in self.testwords:
            print(word)
            nn_words = self.model.get_similar_words(word, n)
            for w, sim in nn_words.items():
                print(f"{w} ({sim:.3})", end=' ')
            print('\n')
    '''