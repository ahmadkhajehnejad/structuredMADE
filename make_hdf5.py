
import config
from dataset import make_hdf5

if __name__ == '__main__':
    args = {'data_name': config.data_name, 'train_size': config.train_size, 'valid_size': config.validation_size,
            'test_size': config.test_size}
    if args['data_name'] == 'grid':
        args['width'] = config.width
        args['height'] = config.height
    elif args['data_name'] == 'Boltzmann':
        args['n'] = config.n_boltzmann
        args['m'] = config.m_boltzmann
    elif args['data_name'] == 'k_sparse':
        args['n'] = config.n_of_k_sparse
        args['sparsity_degree'] = config.sparsity_degree
    elif args['data_name'] == 'BayesNet':
        args['n'] = config.n_of_BayesNet
        args['par_num'] = config.par_num_of_BayesNet
    elif args['data_name'].startswith('mnist'):
        args['digit'] = config.digit

    data = make_hdf5( args, config.hdf5_file_name)