import torch
import torch.nn as nn

def eucl_distance(x, y, bias=None):
	'''
		takes x of shape 1 x d or n x d and y of shape n x d and computes the
		pairwise euclidean distance. if bias is not None, add bias.
	'''
	dist_fn = nn.PairwiseDistance(p=2)
	if bias is None:
		return dist_fn(x, y).pow(2)
	else:
		return dist_fn(x, y).pow(2) + bias
