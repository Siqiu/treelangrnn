import torch
import torch.nn as nn
import numpy as np


class LinearRNNCell(nn.RNN):

	def __init__(ninp, nhid, dropout=0):

		super(ELURNNCell, self).__init__(ninp, nhid, 1, dropout=dropout)

	def forward(self, input_, h_0):

		seq_len, bsz, ninp = input_.size()
		in_times_W = torch.functional.linear(input_, self.weights_ih_l0, self.bias_ih_l0)
	
		h = h_0
		output = []
		for t in range(seq_len):

			h_times_U = torch.nn.functional.linear(h, self.weights_hh_l0, self.bias_hh_l0)
			output.append(in_times_W[t] + h_times_U)
			h = output[-1]

		return torch.cat(output, 1), None


class ELURNNCell(nn.RNN):

	def __init__(ninp, nhid, dropout=0, alpha=1.0):

		super(ELURNNCell, self).__init__(ninp, nhid, 1, dropout=dropout)
		self.elu = nn.ELU(alpha=alpha)

	def forward(self, input_, h_0):

		seq_len, bsz, ninp = input_.size()
		in_times_W = torch.functional.linear(input_, self.weights_ih_l0, self.bias_ih_l0)
	
		h = h_0
		output = []
		for t in range(seq_len):

			h_times_U = torch.nn.functional.linear(h, self.weights_hh_l0, self.bias_hh_l0)
			output.append(self.elu(in_times_W[t] + h_times_U))
			h = output[-1]

		return torch.cat(output, 1), None


class DExpRNNCell(nn.RNN):


	def __init__(ninp, nhid, dropout=0, alpha=0.1):

		super(ELURNNCell, self).__init__(ninp, nhid, 1, dropout=dropout)
		self.alpha = alpha

	def dilated_exp(x):
		return (torch.exp(self.alpha * x) - 1) / self.alpha


	def forward(self, input_, h_0):

		seq_len, bsz, ninp = input_.size()
		in_times_W = torch.functional.linear(input_, self.weights_ih_l0, self.bias_ih_l0)
	
		h = h_0
		output = []
		for t in range(seq_len):

			h_times_U = torch.nn.functional.linear(h, self.weights_hh_l0, self.bias_hh_l0)
			output.append(self.dilated_exp(in_times_W[t] + h_times_U))
			h = output[-1]

		return torch.cat(output, 1), None

