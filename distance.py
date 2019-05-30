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


def dot_distance(x, y, bias=None):
	'''
		takes x of shape 1 x d or n x d and y of shape n x d and computes
		the dot product of x with y. 
	'''
	sim_fn = torch.nn.functional.linear
	if x.size(0) > 1:
		return torch.diag(-sim_fn(x, y, bias=bias))
	else:
		return -sim_fn(x, y, bias=bias)

def cone_distance(x, y, bias=None):
    
    # x of shape 1 x hsz
    # y of shape n x hsz
    
    x_norm = x.norm(dim=1)                       # shape 1
    y_norm = y.norm(dim=1)          # shape n
    xy_norm = (x-y).norm(dim=1)     # shape 

    # calculate the angle between x and 
    top = y_norm.pow(2) - x_norm.pow(2) - xy_norm.pow(2)
    btm = 2 * x_norm * xy_norm + 1e-5
    arg = 1e-5 + (top / btm)

    # clip s.t. input is clean
    arg = torch.clamp(arg, min=-1, max=1)
    return torch.acos(arg)


