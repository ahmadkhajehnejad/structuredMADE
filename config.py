hdf5_file_name = 'grid_tr1000_val250_test5000.hdf5'
masks_file = 'masks.pkl' # 'masks_Mr_100.pkl'
num_of_all_masks = 10
num_of_hlayer = 2
hlayer_size = 1200 # 3000 # 800 # 3500 # 
random_dimensions_order = 'grid' # True # False # 'grid_from_center' # 
direct_links = True # 'Full' #  
algorithm = 'Q_restricted' # 'min_related' # 'orig' #  
train_size = 300
validation_size = 75
test_size = 5000
data_name = 'mnistdps4' # 'cifar10' # 'mnist' # 'grid' # 'binarized_mnist' # 'ocrdp1' # 'k_sparse'
learn_alpha = False
random_data = True # False # 
use_best_validated_weights = True
fast_train = False # True # 
logReg_pretrain = False # True #         
from keras import optimizers

scale_negOne_to_posOne = False # True # 

use_cnn = False # True #   
num_resnet_blocks = 30 # 5 # 
size_resnet_block = 5
stateful_cnn = True

patch_MADE = -1

component_form = 'logistic' # 'Gaussian' # 
num_mixture_components = 3 # 1 # 
#min_var = 2.735619835232257e-06 #.1
min_logScale = -7.0
robust = False
#logistic_cdf_inf = 3 # 10
use_uniform_noise_for_pmf = False # True
# num_noisy_samples_per_sample = 1
use_logit_preprocess = False # True # 
logit_scale = 0.99

AE_adam = optimizers.Adam(lr=0.0001, beta_1=0.1)

## General configs
use_multiprocessing = True
generate_samples = False
generated_samples_dir = './generated_samples_patched_bs50_mnist_' + str(train_size) + '_' + algorithm + '_' + str(random_dimensions_order) + '/'
num_of_generated_samples_each_execution = 20
num_of_exec = 3
fit_iter = 1
num_of_epochs = 1000   #max number of epoch if not reaches the ES condition
batch_size = 20 # 50 # 1 # 
test_batch_size = 1000
optimizer = AE_adam
patience = 5
Q_restricted_2_pass = True

## grid configs
if data_name == 'grid':
    width = 4
    height = 4
    graph_size = width * height

## mnist configs
if data_name.startswith('mnist'):
    num_channels = 1
    width = 28 * num_channels
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

# cifar
if data_name.startswith('cifar'):
    num_channels = 3
    width = 32 * num_channels
    height = 32
    graph_size = width * height

##############
