from collections import namedtuple

# container to store settings
SampleConfig = namedtuple("SampleConfig", "nsamples uniform_freq frequencies")

nsamples = 10			# number of negative samples for each positive example
uniform_freq = True		# if true, words are sampled from a uniform distribution
frequencies = None		# None, don't change

sample_config = SampleConfig(nsamples, uniform_freq, frequencies)