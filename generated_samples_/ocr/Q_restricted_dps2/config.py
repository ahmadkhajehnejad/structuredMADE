
use_multiprocessing = True
num_of_all_masks = 10
num_of_hlayer = 2
hlayer_size = 1200
random_dimensions_order = True
direct_links = 'Full'
algorithm = 'orig'
train_size = 1000
validation_size = 1000
test_size = 5000
data_name = 'ocrdps2' # 'mnistdps4' # 'grid' # 'k_sparse'
random_data = False
from keras import optimizers

## General configs

AE_adam = optimizers.Adam(lr=0.0003, beta_1=0.1)

use_multiprocessing = True
generate_samples = True
generated_samples_dir = './generated_samples/'
num_of_generated_samples_each_execution = 100
num_of_exec = 1
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
if data_name.startswith('mnist'):
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

# BayesNet configs
if data_name == 'BayesNet':
    n_of_BayesNet = 100
    par_num_of_BayesNet = 5
    graph_size = n_of_BayesNet

# ocr
if data_name.startswith('ocr'):
    width = 8
    height = 16
    graph_size = width * height
    ocr_characters = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 24, 25]
    # ocr_num_character_samples = [4034, 1284, 2114, 1442, 4955,  921, 2472,  861, 4913,\
    #                             189,  909, 3140, 1602, 5024, 3897, 1377,  341, 2673,\
    #                             1394, 2136, 2562,  664, 520,  413, 1221, 1094]
##############
