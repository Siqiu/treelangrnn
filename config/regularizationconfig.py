from collections import namedtuple

# container to store settings
RegularizationConfig = namedtuple("RegularizationConfig", "dropout dropouth dropouti dropoute wdrop alpha beta wdecay")

# dropout
dropout = 0.
dropouth = 0.
dropouti = 0.
dropoute = 0.
wdrop = 0.

# regularizers
alpha = 0
beta = 0
wdecay = 1.2e-6

reg_config = RegularizationConfig(dropout, dropouth, dropouti, dropoute, wdrop, alpha, beta, wdecay)
