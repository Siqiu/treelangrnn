from collections import namedtuple


# container to store settings
RNNConfig = namedtuple("RNNConfig", "cell_type emsize nhid temp")

# basic rnn settings
cell_type = 'rnn'	# cell type, chose one from [rnn, linear_rnn, relu_rnn, elu_rnn, dexp_rnn, gru]
emsize = 100		# dimension of the word embeddings
nhid = 100			# dimension of the hidden sattes
temp = 1			# temperature parameter in the exponent

# for fixed word embeddings give path
word_embeddings_path = None

rnn_config = RNNConfig(cell_type, emsize, nhid temp, word_embeddings_path)