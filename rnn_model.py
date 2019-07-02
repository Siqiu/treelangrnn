import torch
import torch.nn as nn
import numpy as np

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

from sample import NegativeSampler

from utils.utils import repackage_hidden

from eucl_distance.distance import eucl_distance, dot_distance
from poinc_distance.poinc_distance import poinc_distance
from activation import log_softmax, log_sigmoid

from threshold import hard_threshold, soft_threshold1, soft_threshold2, DynamicThreshold

class RNNModel(nn.Module):
    """Container module with an encoder and a recurrent module."""

    def __init__(self, ntoken, ninp, nhid, beta=10, bias=True, dist_fn='eucl', mode='rnn', sampling=None, dropouts=None, regularizers=None, threshold=None):

        initrange = 0.1
        super(RNNModel, self).__init__()
        ntoken = ntoken

        # initialize dropouts
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouts.dropouti)
        self.hdrop = nn.Dropout(dropouts.dropouth)
        self.drop = nn.Dropout(dropouts.dropout)
       
        self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder = self.init_weights(self.encoder, initrange)

        self.mode = mode
        if self.mode == 'rnn':
            self.rnn = torch.nn.RNN(ninp, nhid, 1, dropout=0)#, nonlinearity='relu')
        elif self.mode =='gru':
            self.rnn = torch.nn.GRU(ninp, nhid, 1, dropout=0)

        self.rnn = WeightDrop(self.rnn, ['weight_hh_l0'], dropout=dropouts.wdrop)
        print(self.rnn)

        # initialize bias
        if bias:
            self.decoder = nn.Linear(nhid, ntoken)
            self.decoder = self.init_weights(self.decoder, 0.1)
            self.bias = self.decoder.bias
        else: self.bias = None

        # store input arguments
        self.ninp = ninp
        self.nhid = nhid
        self.dropout = dropouts.dropout
        self.dropouti = dropouts.dropouti
        self.dropouth = dropouts.dropouth
        self.dropoute = dropouts.dropoute
        self.wdrop = dropouts.wdrop

        # regularizers
        self.regularizers = regularizers

        # nonlinearity needs to be the same as for RNN!
        self.nonlinearity = nn.Tanh()#nn.ReLU()
        self.activation = log_softmax
        self.ntoken = ntoken    # number of tokens
        self.beta = beta        # temperature

        self.nsamples = sampling.nsamples
        print(sampling.frequencies)
        self.sampler = NegativeSampler(self.nsamples, torch.ones(self.ntoken) if sampling.frequencies is None else sampling.frequencies)

        # set distance function
        if dist_fn == 'eucl': self.dist_fn = eucl_distance
        elif dist_fn == 'dot': self.dist_fn = dot_distance
        elif dist_fn == 'poinc': self.dist_fn = poinc_distance
        else: self.dist_fn = None

        if threshold.method == 'hard': self.threshold = hard_threshold
        elif threshold.method == 'soft1': self.threshold = soft_threshold1
        elif threshold.method == 'soft2': self.threshold = soft_threshold2
        elif threshold.method == 'dynamic': self.threshold = DynamicThreshold(nhid, threshold.nhid, threshold.nlayers, threshold.temp)
        else: self.threshold = None
        self.threshold_method = threshold.method

        self.radius = threshold.radius
        self.inf = 1e5
        
      

    def init_weights(self, module, initrange=0.1):
        module.weight.data.uniform_(-initrange, initrange)
        return module

    def _apply_threshold(self, d, h):
        '''
            d: pairwise distances between h and h_+
            h: initial hidden states h
        '''
        if self.threshold_method == 'dynamic':
            d, r = self.threshold(d, h, self.inf)
            return d
        else:
            return self.threshold(d, self.radius, self.inf)

    def _apply_temperature(self, d):
        return self.beta * d

    def _apply_bias(self, d, b):
        return d + b

    def _forward(self, words_times_W, hiddens_times_U, hidden=None):

        tanh, sigmoid = nn.Tanh(), nn.Sigmoid()

        if self.mode == 'rnn':
            output = tanh(words_times_W + hiddens_times_U)

        elif self.mode == 'gru':
            _ir = sigmoid(words_times_W[:,:self.nhid] + hiddens_times_U[:,:self.nhid])
            _iz = sigmoid(words_times_W[:,self.nhid:2*self.nhid] + hiddens_times_U[:,self.nhid:2*self.nhid])
            _in = tanh(words_times_W[:,2*self.nhid:] + _ir * hiddens_times_U[:,2*self.nhid:])
            output = (1 - _iz) * _in + _iz * hidden

        return output

    def forward(self, data, binary, hidden):

        # get batch size and sequence length
        seq_len, bsz = data.size()

        emb = embedded_dropout(self.encoder, data, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output, new_hidden = self.rnn(emb, hidden)          # apply single layer rnn
        raw_output = self.lockdrop(raw_output, self.dropout)    # seq_len x bsz x nhid
        raw_output = raw_output.view(seq_len, bsz, -1)          # reshape for concat
        raw_output = torch.cat((hidden, raw_output), 0)         # concatenate initial hidden state

        # initialize loss w/ positive terms
        # compute distances between consecutive hidden states
        d_pos = (raw_output[1:] - raw_output[:-1]).norm(dim=2).pow(2)

        if not self.threshold is None:
            pass#d_pos = self._apply_threshold(d_pos, raw_output[:-1])

        d_pos = self._apply_temperature(d_pos)

        if not self.bias is None:
            d_pos = self._apply_bias(d_pos, self.bias[data])

        # hiddens used for negative sampling are all except last
        raw_output = raw_output[:-1].view(seq_len*bsz, -1)

        # x stores the positive samples at index 0 and the negative ones a 1:nsamples+1
        x = torch.zeros(1 + self.nsamples, seq_len*bsz).cuda()
        x[0] = -d_pos.view(seq_len * bsz)
        
        # process negative samples
        samples = self.sampler(bsz, seq_len)    # (nsamples x bsz x seq_len)
        samples_emb = embedded_dropout(self.encoder, samples, dropout=self.dropoute if self.training else 0)
        samples_emb = self.lockdrop(samples_emb, self.dropouti)

        # only one layer for the moment
        weights_ih, bias_ih = self.rnn.module.weight_ih_l0, self.rnn.module.bias_ih_l0  
        weights_hh, bias_hh = self.rnn.module.weight_hh_l0, self.rnn.module.bias_hh_l0

        # reshape samples for indexing and precompute the inputs to nonlinearity
        samples = samples.view(self.nsamples, bsz*seq_len)
        samples_times_W = torch.nn.functional.linear(samples_emb, weights_ih, bias_ih).view(self.nsamples, bsz*seq_len, -1)
        hiddens_times_U = torch.nn.functional.linear(raw_output, weights_hh, bias_hh)
        
        # iterate over samples to update loss
        for i in range(self.nsamples):

            # compute output of negative samples
            output = self._forward(samples_times_W[i], hiddens_times_U, raw_output)
            output = self.lockdrop(output.view(1, output.size(0), -1), self.dropout)
            output = output[0]

            # compute loss term
            d_neg = self.dist_fn(raw_output, output)

            if not self.threshold is None:
                pass#d_neg = self._apply_threshold(d_neg, raw_output)

            d_neg = self._apply_temperature(d_neg)

            if not self.bias is None:
                d_neg = self._apply_bias(d_neg, self.bias[samples[i]])
        
            x[i+1] = -d_neg

        softmaxed = -torch.nn.functional.log_softmax(x, dim=0)[0]
        softmax_mapped = softmaxed.view(seq_len, bsz) * binary
        loss = softmax_mapped.mean()
        
        # apply regularizer for bias
        if self.regularizers.bias > 0:
            loss = loss + (0 if self.bias is None else self.regularizers.bias * torch.norm(self.bias).pow(2))

        return loss


    def evaluate(self, data, eos_tokens=None, dump_hiddens=False):

        # get weights and compute WX for all words
        weights_ih, bias_ih = self.rnn.module.weight_ih_l0, self.rnn.module.bias_ih_l0  # only one layer for the moment
        weights_hh, bias_hh = self.rnn.module.weight_hh_l0, self.rnn.module.bias_hh_l0

        all_words = torch.LongTensor([i for i in range(self.ntoken)]).cuda()
        all_words = embedded_dropout(self.encoder, all_words, dropout=self.dropoute if self.training else 0)

        all_words_times_W = torch.nn.functional.linear(all_words, weights_ih, bias_ih)

        # iterate over data set and compute loss
        total_loss, hidden = 0, self.init_hidden(1)
        i = 0

        entropy, hiddens, all_hiddens = [], [], []
        while i < data.size(0):

            hidden_times_U = torch.nn.functional.linear(hidden[0].repeat(self.ntoken, 1), weights_hh, bias_hh)
            output = self._forward(all_words_times_W, hidden_times_U, hidden[0].repeat(self.ntoken, 1))

            if dump_hiddens: hiddens.append(output[data[i]].data.cpu().numpy())

            distance = self.dist_fn(hidden[0], output)
            if not self.threshold is None:
                distance = self._apply_threshold(distance, hidden[0])

            distance = self._apply_temperature(distance)

            if not self.bias is None:
                distance = self._apply_bias(distance, self.bias)
        
            softmaxed = torch.nn.functional.log_softmax(-distance, dim=0)
            raw_loss = -softmaxed[data[i]].item()

            total_loss += raw_loss / data.size(0)
            entropy.append(raw_loss)

            if not eos_tokens is None and data[i].data.cpu().numpy()[0] in eos_tokens:
                hidden = self.init_hidden(1)
                if dump_hiddens:
                    all_hiddens.append(hiddens)
                    hiddens = []
            else:
                hidden = output[data[i]].view(1, 1, -1)
            hidden = repackage_hidden(hidden)

            i = i + 1

        all_hiddens = all_hiddens if not eos_tokens is None else hiddens
        
        if dump_hiddens:
            return total_loss, np.array(entropy), all_hiddens
        else:
            return total_loss, np.array(entropy)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new(1, bsz, self.nhid).zero_()
