import numpy as np
import config

def _get_data_from_file(args):
    
    with np.load(args['data_file']) as parameters:
        
        all_outcomes = parameters['all_outcomes']
        prob_of_outcomes = parameters['prob_of_outcomes']
        
        data = parameters['train_data']
        data_probs = parameters['train_data_probs']
        
        train_data = data[0:args['train_size']]
        valid_data = data[args['train_size']:(args['train_size']+args['valid_size'])]
        
        
        if not(data_probs is None):
            train_data_probs = data_probs[0:args['train_size']]
            valid_data_probs = data_probs[args['train_size']:(args['train_size']+args['valid_size'])]
        else:
            train_data_probs = None
            valid_data_probs = None
        
        if args['test_size'] == 'FULL_TEST':
            test_data = all_outcomes
            test_data_probs = prob_of_outcomes
        else:
            data = parameters['test_data']
            data_probs = parameters['test_data_probs']
            test_data = data[0:args['test_size']]
            if not(data_probs is None):
                test_data_probs = data_probs[0:args['test_size']]
            else:
                test_data_probs = None
            
    return {'train_data' : train_data,
            'train_data_probs' : train_data_probs,
            'valid_data' : valid_data,
            'valid_data_probs' : valid_data_probs,
            'test_data' : test_data,
            'test_data_probs' : test_data_probs}

def get_data(args):
    if args['data_name'] == 'grid':
        args['data_file'] = 'datasets/grid' + str(args['height']) + 'by' + str(args['width']) + '.npz'
        return _get_data_from_file(args)
    elif args['data_name'] == 'Boltzmann':
        args['data_file'] = 'datasets/Boltzman_' + str(args['n']) + '&' + str(args['m']) + '.npz'
        return _get_data_from_file(args)
    elif args['data_name'] == 'mnist':
        args['data_file'] = 'datasets/binary_mnist_'+ str(args['digit']) + '.npz'
        return _get_data_from_file(args)
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
        graph_size = 14 * 14
        adj = np.zeros([graph_size, graph_size])
        for r in range(0, 14):
            for c in range(0, 14):
                jj = r*14 + c
                if c > 0:
                    adj[jj-1][jj] = adj[jj][jj-1] = 1
                if r > 0:
                    adj[jj-14][jj] = adj[jj][jj-14] = 1
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
