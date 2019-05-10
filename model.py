import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

from sample import NegativeSampler

class RNNModel(nn.Module):
    """Container module with an encoder and a recurrent module."""

    def __init__(self, ntoken, ninp, nhid, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, nsamples=10, temperature=65, frequencies=None, clip_dist=0.01):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnn = torch.nn.RNN(ninp, nhid, 1, dropout=0)
        print(self.rnn)

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute

        self.nonlinearity = nn.Tanh()
        self.eps = 1e-6
        self.nsamples = nsamples
        self.temp = temperature
        self.ntoken = ntoken
        self.clip_dist = clip_dist

        print(frequencies)
        self.sampler = NegativeSampler(self.nsamples, torch.ones(self.ntoken) if frequencies is None else frequencies)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, data, return_output=False):

        # need this later on
        dist_fn = nn.PairwiseDistance(p=2)

        # get batch size and sequence length
        seq_len, bsz = data.size()

        # process positive samples
        hidden = self.init_hidden(bsz)      # bsz x nhid

        emb = embedded_dropout(self.encoder, data, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output, new_hidden = self.rnn(emb, hidden)          # apply single layer rnn
        raw_output = self.lockdrop(raw_output, self.dropout)    # seq_len x bsz x nhid
        raw_output = raw_output.view(seq_len, bsz, -1)
        raw_output = torch.cat((hidden, raw_output), 0)

        # initialize loss w/ positive terms
        pos_sample_distances = [self.temp * dist_fn(raw_output[i], raw_output[i+1]).pow(2) for i in range(seq_len)]
        raw_output = raw_output[:-1].view(seq_len*bsz, -1)

        # we want positive terms in the sum as well
        sum_of_exp = torch.zeros(seq_len*bsz).cuda()
        #for i in range(seq_len):
        #    sum_of_exp[i*bsz:(i+1)*bsz] = torch.exp(-pos_sample_distances[i])
        
        # init loss
        loss = sum(pos_sample_distances).sum() / len(pos_sample_distances)
        #loss = 0
           
        # process negative samples
        samples = self.sampler(bsz, seq_len)    # (nsamples x bsz x seq_len)

        samples_emb = embedded_dropout(self.encoder, samples, dropout=self.dropoute if self.training else 0)
        #samples_emb = self.lockdrop(samples_emb, self.dropouti)

        weights_ih, bias_ih = self.rnn.weight_ih_l0, self.rnn.bias_ih_l0  # only one layer for the moment
        weights_hh, bias_hh = self.rnn.weight_hh_l0, self.rnn.bias_hh_l0

        samples_times_W = torch.nn.functional.linear(samples_emb, weights_ih, bias_ih).view(self.nsamples, bsz*seq_len, -1)
        hiddens_times_U = torch.nn.functional.linear(raw_output, weights_hh, bias_hh)

        # iterate over samples to update loss
        for i in range(self.nsamples):

            # compute output of negative samples
            output = self.nonlinearity(samples_times_W[i] + hiddens_times_U)

            # compute loss term
            distance = dist_fn(raw_output, output).pow(2)
            if self.clip_dist:
                distance = torch.clamp(distance, 0, self.clip_dist)
                print('clamping')

            sum_of_exp = sum_of_exp + torch.exp(-distance) / len(distance)

        loss = loss + torch.log(sum_of_exp + self.eps).sum()
        
        return loss


    def train_crossentropy(self, data, eos_tokens):

        dist_fn = nn.PairwiseDistance(p=2)

        # get weights and compute WX for all words
        weights_ih, bias_ih = self.rnn.weight_ih_l0, self.rnn.bias_ih_l0  # only one layer for the moment
        weights_hh, bias_hh = self.rnn.weight_hh_l0, self.rnn.bias_hh_l0

        all_words = torch.LongTensor([i for i in range(self.ntoken)]).cuda()
        all_words = embedded_dropout(self.encoder, all_words, dropout=self.dropoute if self.training else 0)

        all_words_times_W = torch.nn.functional.linear(all_words, weights_ih, bias_ih)

        # iterate over data set and compute loss
        total_loss, hidden = 0, self.init_hidden(1)
        for i in range(data.size(0)):

            hidden_times_U = torch.nn.functional.linear(hidden[0].repeat(self.ntoken, 1), weights_hh, bias_hh)
            output = self.nonlinearity(all_words_times_W + hidden_times_U)

            distance = dist_fn(hidden[0], output).pow(2)
            softmaxed = torch.nn.functional.log_softmax(-self.temp * distance, dim=0)
            raw_loss = -softmaxed[data[i]]
            total_loss += raw_loss / data.size(0)

            if data[i].data.cpu().numpy()[0] in eos_tokens:
                hidden = self.init_hidden(1)
            else:
                hidden = output[data[i]].view(1, 1, -1)

        return total_loss


    def evaluate(self, data, eos_tokens):

        dist_fn = nn.PairwiseDistance(p=2)

        # get weights and compute WX for all words
        weights_ih, bias_ih = self.rnn.weight_ih_l0, self.rnn.bias_ih_l0  # only one layer for the moment
        weights_hh, bias_hh = self.rnn.weight_hh_l0, self.rnn.bias_hh_l0

        all_words = torch.LongTensor([i for i in range(self.ntoken)]).cuda()
        all_words = embedded_dropout(self.encoder, all_words, dropout=self.dropoute if self.training else 0)

        all_words_times_W = torch.nn.functional.linear(all_words, weights_ih, bias_ih)

        # iterate over data set and compute loss
        total_loss, hidden = 0, self.init_hidden(1)
        i = 0
        seq = []
        while i < data.size(0):

            hidden_times_U = torch.nn.functional.linear(hidden[0].repeat(self.ntoken, 1), weights_hh, bias_hh)
            output = self.nonlinearity(all_words_times_W + hidden_times_U)

            distance = dist_fn(hidden[0], output).pow(2)
            softmaxed = torch.nn.functional.log_softmax(-self.temp * distance, dim=0)
            raw_loss = -softmaxed[data[i]].item()

            total_loss += raw_loss / data.size(0)

            seq.append(data[i])

            if data[i].data.cpu().numpy()[0] in eos_tokens:
                hidden = self.init_hidden(1)
                seq = []
            else:
                hidden = output[data[i]].view(1, 1, -1)

            i = i + 1

        return total_loss


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new(1, bsz, self.nhid).zero_()
