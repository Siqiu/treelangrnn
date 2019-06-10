import torch
import torch.nn as nn

def hard_threshold(d, r, inf=1e4):
	# set every distance larger than r to inf
	d[d >= r] = inf
	return d

def soft_threshold1(d, r, inf=1e4):
	# continuous at d = r
	# f(d,r) = d if d < r else r * exp(d - r)
	idxs = d < r
	dnew = r * torch.exp(d - r)
	dnew[idxs] = d[idxs]
	return torch.clamp(dnew, max=inf)


def soft_threshold2(d, r, inf=1e4):
	# continuous at d = r 
	# f(d,r) = d if d < r else r + exp(d-r) - 1
	idxs = d < r
	dnew = r + torch.exp(10*(d - r)) - 1
	dnew[idxs] = d[idxs]
	return torch.clamp(dnew, max=inf)


class DynamicThreshold(nn.Module):

	def __init__(self, nin, nhid, nlayers):

		super(DynamicThreshold, self).__init__()

		self.nin = nin
		self.nhid = nhid
		self.nlayers = nlayers

		# build neural net here
		linears = [nn.Linear(nin, nhid) if l == 0 else nn.Linear(nhid, nhid) for l in range(nlayers - 1)]
		relus = [nn.ReLU() for l in range(nlayers - 1)]
		modules = [mod for pair in zip(linears, relus) for mod in pair] + [nn.Linear(nin, 1) if nlayers == 1 else nn.Linear(nhid, 1)]

		self.net = nn.Sequential(*modules)

	def forward(self, d, hiddens, inf=1e4):

		# get r from neural net
		r = self.net(hiddens).pow(2)
		print(r)
		if r.size(0) > 1:
			r = r.view(d.size())

		idxs = (d < r).view(d.size())
		dnew = r * torch.exp(d - r)
		print(dnew.size(), d.size(), idxs.size(), r.size() )
		dnew[idxs] = d[idxs]
		return soft_threshold1(d, r, inf)
