import torch
import torch.nn as nn
import numpy as np


class DynamicRNNCell(nn.RNN):

	def __init__(self, ninp, nhid, dropout=0, nlayers=4):

		super(DynamicRNNCell, self).__init__(ninp, nhid, 1, dropout=dropout)
		self.ninp, self.nhid = ninp, nhid

		# build neural net here
		linears = [nn.Linear(ninp, nhid) if l == 0 else nn.Linear(nhid, nhid) for l in range(nlayers)]
		relus = [nn.Tanh() for l in range(nlayers)]
		modules = [mod for pair in zip(linears, relus) for mod in pair]
		self.net = nn.Sequential(*modules)

	def forward(self, input_, h_0):

		# assume nhid == ninp!!!
		u, s, v = torch.svd(self.weight_ih_l0)
		v = v.t()

		seq_len, ninp = input_.size()
	
		h = h_0
		output = []
		for t in range(seq_len):

			s = self.net(h).pow(2)
			weight_ih_l0 = torch.mm(u, torch.mm(torch.diag(s[0][0]), v))

			in_times_W = torch.nn.functional.linear(input_[t], weight_ih_l0, self.bias_ih_l0)
			h_times_U = torch.nn.functional.linear(h, self.weight_hh_l0, self.bias_hh_l0)
			output.append(torch.nn.functional.tanh(in_times_W + h_times_U))
			h = output[-1]

		return torch.cat(output, 1)

