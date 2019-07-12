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

		self.U, self.S, self.V = torch.svd(self.weight_ih_l0)

	def _in_times_W(self, input_, h):

		_in_times_V = torch.nn.functional.linear.(input_, self.V.t())
		S = self.net(h)
		_in_times_SV = S * _in_times_V
		_in_times_USV = torch.nn.functional.linear(_in_times_SV, self.U)
		return _in_times_USV

	def forward(self, input_, h_0, do_svd=True):

		# assume nhid == ninp!!!
		if do_svd:
			self.U, self.S, self.V = torch.svd(self.weight_ih_l0)

		seq_len, ninp = input_.size()
	
		h = h_0
		output = []
		for t in range(seq_len):

			in_times_W = self._in_times_W(input_[t], h)
			h_times_U = torch.nn.functional.linear(h, self.weight_hh_l0, self.bias_hh_l0)
			output.append(torch.nn.functional.tanh(in_times_W + h_times_U))
			h = output[-1]

		return torch.cat(output, 1)

