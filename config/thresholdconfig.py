from collections import namedtuple

# container to store settings
Threshold = namedtuple("Threshold", "method mode temp min_radius max_radius decrease_radius nlayers nhid lr")

threshold_method = 'dynamic'
threshold_mode = 'both'
threshold_temp = 1

# this is for static radius only
threshold_min_radius = 1
threshold_max_radius = 5
threshold_decrease_radius = 0.	# if 0. -> don't decrease radius but use max_radius

# this is for dynamic radii only
threshold_nlayers = 8
threshold_nhid = 150
threshold_lr = 0.1				# if 0. -> no separate learning rate

threshold_config = Threshold(args.threshold_method, args.threshold_mode, args.threshold_temp,
							args.threshold_min_radius, args.threshold_max_radius, args.treshold_decrease_radius,
							args.threshold_nlayers, args.threshold_nhid, args.threshold_lr) if not args.threshold_method == 'none' else None
