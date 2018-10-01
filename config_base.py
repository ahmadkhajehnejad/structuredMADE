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
width = 4
height = 4
related_size = width

## mnist configs
digit = 6


# Boltsmann configs
n_boltzmann = 10
m_boltzmann = 10

##############

if data_name == 'grid':
    graph_size = width * height
elif data_name == 'mnist':
    graph_size = 14*14
elif data_name == 'Boltzmann':
    graph_size = n_boltzmann + m_boltzmann
