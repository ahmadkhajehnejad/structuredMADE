import numpy as np
import config

def _get_Ising_data(args):
    
    with np.load(args['parameters_file']) as parameters:
        all_outcomes = parameters['all_outcomes']
        prob_of_outcomes = parameters['prob_of_outcomes']
        cum_probs = parameters['cum_probs']
    

    p = np.random.uniform(0,1,args['train_size'])
    ind = np.searchsorted(cum_probs, p)
    train_data = all_outcomes[ind,:]
    train_data_probs = prob_of_outcomes[ind]           

    p = np.random.uniform(0,1,args['validation_size'])
    ind = np.searchsorted(cum_probs, p)
    valid_data = all_outcomes[ind,:]
    valid_data_probs = prob_of_outcomes[ind]        
    
    if args['test_size'] == 'FULL_TEST':
        test_data = all_outcomes.copy()
        test_data_probs = prob_of_outcomes.copy()
    else:
        p = np.random.uniform(0,1,args['test_size'])
        ind = np.searchsorted(cum_probs, p)
        test_data = all_outcomes[ind,:]
        test_data_probs = prob_of_outcomes[ind]

    #test_data = all_outcomes[prob_of_outcomes > 0][:]
    #test_data_probs = prob_of_outcomes[prob_of_outcomes > 0]
    
    return {'train_data' : train_data,
            'train_data_probs' : train_data_probs,
            'valid_data' : valid_data,
            'valid_data_probs' : valid_data_probs,
            'test_data' : test_data,
            'test_data_probs' : test_data_probs}

def _get_mnist_data(args):

    with np.load('datasets/binary_mnist_1.npz') as dataset:
        data = np.copy(dataset['train_data'])
        np.random.shuffle(data)
        train_data = data[0:train_length][:]
        np.random.shuffle(data)
        valid_data = data[0:valid_length][:]
        data = np.copy(dataset['test_data'])
        np.random.shuffle(data)
        test_data = data[0:test_length][:]
    train_data_probs = None
    valid_data_probs = None
    test_data_probs = None
    return {'train_data' : train_data,
            'train_data_probs' : train_data_probs,
            'valid_data' : valid_data,
            'valid_data_probs' : valid_data_probs,
            'test_data' : test_data,
            'test_data_probs' : test_data_probs}

def get_data(args):
    if config.data_name == 'grid':
        args['parameters_file'] = 'dataset_parameters/grid' + str(config.height) + 'by' + str(config.width) + '_parameters.npz'
        return _get_Ising_data(args)
    elif config.data_name == 'Boltzmann':
        args['parameters_file'] = 'dataset_parameters/Boltzman_' + str(config.n_boltzmann) + ',' + str(config.m_boltzmann) + '_parameters.npz'
        return _get_Ising_data(args)
    elif config.data_name == 'mnist':
        return _get_mnist_data(args)
    else:
        return None
                

def get_data_structure():
    parameters = dict()
    if config.data_name == 'grid':
        graph_size = config.height * config.width
        adj = np.zeros([graph_size, graph_size])
        for r in range(0, config.height):
            for c in range(0, config.width):
                jj = r*config.width + c
                if c > 0:
                    adj[jj-1][jj] = adj[jj][jj-1] = 1
                if r > 0:
                    adj[jj-config.width][jj] = adj[jj][jj-config.width] = 1
        parameters['adjacency_matrix'] = adj
        
    elif config.data_name == 'mnist':
        graph_size = 28 * 28
        adj = np.zeros([graph_size, graph_size])
        for r in range(0, 28):
            for c in range(0, 28):
                jj = r*28 + c
                if c > 0:
                    adj[jj-1][jj] = adj[jj][jj-1] = 1
                if r > 0:
                    adj[jj-28][jj] = adj[jj][jj-28] = 1
        parameters['adjacency_matrix'] = adj
        
    elif config.data_name == 'Boltzmann':
        n = config.n_boltzmann
        m = config.m_boltzmann
        graph_size = n + m
        adj = np.zeros([graph_size, graph_size])
        for i in range(n):
            for j in range(m):
                adj[i,n+j] = adj[n+j,i] = 1
        parameters['adjacency_matrix'] = adj
    return parameters
