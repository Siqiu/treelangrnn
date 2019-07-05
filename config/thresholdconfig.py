from collections import namedtuple

# container to store settings
ThresholdConfig = namedtuple("ThresholdConfig", "func mode temp min_radius max_radius decrease_radius nlayers nhid lr")

threshold_func = 'soft2'	# threshold function, chose from [hard, soft1, soft2, dynamic]
threshold_mode = 'both'		# threshold mode, chose from [none, both, train, eval] (indicates when thresholding is applied)
threshold_temp = 1			# make threshold steeper by increasing this

# this is for static radius only
threshold_min_radius = 1	# if threshold is decreased over time we need a min radius	
threshold_max_radius = 5	# if threshold is decreased over time this is the starting value, otherwise this is the fixed radius
threshold_decrease_radius = 0.	# if 0. -> don't decrease radius but use max_radius

# this is for dynamic radii only
threshold_nlayers = 8		# number of layers in the FFNN to compute the radius for each hidden state
threshold_nhid = 150		# dimension of hidden states of FFNN
threshold_lr = 10			# if 0. -> no separate learning rate

threshold_config = ThresholdConfig(threshold_method, threshold_mode, threshold_temp,
							threshold_min_radius, threshold_max_radius, threshold_decrease_radius,
							threshold_nlayers, threshold_nhid, threshold_lr) if not threshold_method == 'none' else None
