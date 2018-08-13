from keras import optimizers


## General configs

AE_adam = optimizers.Adam(lr=0.0003, beta_1=0.1)
num_of_exec = 10
num_of_all_masks = 10
num_of_hlayer = 2
hlayer_size = 225
fit_iter = 1
num_of_epochs = 2000   #max number of epoch if not reaches the ES condition
batch_size = 50
optimizer = AE_adam
patience = 20

train_size = 200
validation_size = 50
test_size = 5000

algorithm = 'Q_restricted'#'orig'#'min_related'#'ensemble_Q_restricted_and_orig'#

data_name = 'rcv1'#'mnist'#'k_sparse'#'min_related'#'grid'#'mnist'#'Boltzmann'

## grid configs
width = 4
height = 4
related_size = width

## mnist configs
digit = 6


# Boltsmann configs
n_boltzmann = 10
m_boltzmann = 10

# k_sparse
n_of_k_sparse = 20
sparsity_degree = 3

##############

if data_name == 'grid':
    graph_size = width * height
elif data_name == 'mnist':
    graph_size = 14*14
elif data_name == 'Boltzmann':
    graph_size = n_boltzmann + m_boltzmann
elif data_name == 'k_sparse':
    graph_size = n_of_k_sparse
elif data_name == 'rcv1':
    graph_size = 150
