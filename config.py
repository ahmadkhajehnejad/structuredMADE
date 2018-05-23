from keras import optimizers


## General configs

AE_adam = optimizers.Adam(lr=0.001, beta_1=0.1)
num_of_exec = 10
num_of_all_masks = 10
num_of_hlayer = 2
hlayer_size = 30
fit_iter = 1
num_of_epochs = 2000   #max number of epoch if not reaches the ES condition
batch_size = 50
optimizer = AE_adam
patience = 30

train_size = 1000
validation_size = 250
test_size = 65536

algorithm = 'Q_restricted'#'orig'#'min_related'

data_name = 'Boltzmann'

## grid configs
width = 4
height = 4
related_size = width

## mnist configs



# Boltsmann configs
n_boltzmann = 10
m_boltzmann = 10

##############

if data_name == 'grid':
    graph_size = width * height
elif data_name == 'mnist':
    graph_size = 28*28
elif data_name == 'Boltzmann':
    graph_size = n_boltzmann + m_boltzmann
