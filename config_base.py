from keras import optimizers


## General configs

AE_adam = optimizers.Adam(lr=0.0003, beta_1=0.1)

num_of_exec = 5
fit_iter = 1
num_of_epochs = 2000   #max number of epoch if not reaches the ES condition
batch_size = 50
optimizer = AE_adam
patience = 2 * num_of_all_masks
Q_restricted_2_pass = True

## grid configs
if data_name == 'grid':
    width = 4
    height = 4
    graph_size = width * height

## mnist configs
if data_name == 'mnist':
    width = 28
    height = 28
    digit = 'All'
    graph_size = width * height

# Boltsmann configs
if data_name == 'Boltzmann':
    n_boltzmann = 10
    m_boltzmann = 10
    graph_size = n_boltzmann + m_boltzmann
    
# k_sparse configs
if data_name == 'k_sparse':
    n_of_k_sparse = 20
    sparsity_degree = 3
    graph_size = n_of_k_sparse
    
##############
