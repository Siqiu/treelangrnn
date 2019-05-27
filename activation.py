import torch
import torch.nn as nn

def log_sigmoid(x):
	'''
		input: x of shape (nsamples + 1) x n
		output: loss (i.e. mean of pos - neg)
	'''
	y = torch.nn.functional.logsigmoid(x)
	return (-y[0] + y[1:].sum(0)).mean()

def log_softmax(x):
	'''
		input: x of shape (nsamples + 1) x n
		output: loss (i.e. mean of pos - neg)
	'''
	return -torch.nn.functional.log_softmax(x, dim=0)[0].mean()

'''
class LogSigmoidNS(nn.Module):

	def __init__(self):
		super(LogSigmoidNS, self).__init__()
		self.activation = nn.LogSigmoid()

	def forward(self, x):
		'''
			input: x of shape (nsamples + 1) x n
			output: loss (i.e. mean of pos - neg)
		'''
		y = self.activation(x)
		return (-y[0] + y[1:].sum(0)).mean()#-(y[0] - y[1:].sum(0)).mean()


class LogSoftmaxNS(nn.Module):

	def __init__(self):
		super(LogSoftmaxNS, self).__init__()
		self.activation = nn.LogSoftmax(dim=0)

	def forward(self, x):


		# the positive samples are at index 0 -> return mean logsoftmax
		return -self.activation(x)[0].mean()

class Simple(nn.Module):

	def __init__(self):
		super(Simple, self).__init__()

	def forward(self, x):
		return -(x[0] -x[1:].sum(0)).mean()
'''
