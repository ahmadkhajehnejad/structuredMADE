import numpy as np
import config

def _get_Ising_data(args):
    
    with np.load(args['parameters_file']) as parameters:
        w_v = parameters['vertex_parameters']
        w_e = parameters['edge_parameters']
    graph_size = len(w_v)
        
    all_outcomes = np.ndarray(shape=(2**graph_size, graph_size), dtype=np.float32)
    prob_of_outcomes = np.ndarray(shape=(2**graph_size), dtype=np.float32)
                
    
    for k in range(2**graph_size):
        str_samp = ('{0:0' + str(graph_size) + 'b}').format(k)
        asarr_samp = [int(d) for d in str_samp]
        all_outcomes[k][:] = asarr_samp
        energy = -np.inner(w_v, asarr_samp)
        energy -= np.matmul(np.array(asarr_samp).reshape([1,-1]), np.matmul(w_e, np.array(asarr_samp).reshape([-1,1])))
        #energy = np.inner(w_v, asarr_samp)
        #energy += np.matmul(np.array(asarr_samp).reshape([1,-1]), np.matmul(w_e, np.array(asarr_samp).reshape([-1,1])))
        p = np.exp(energy)
        prob_of_outcomes[k] = p
    
    sum_prob = sum(prob_of_outcomes)
    prob_of_outcomes = np.divide(prob_of_outcomes, sum_prob)
    
    cum_probs = []
    s = 0
    for x in prob_of_outcomes:
        s = s + x
        cum_probs.append(s)
    cum_probs[-1] = 1.
    
    
    train_data = np.ndarray(shape=(args['train_size'], graph_size), dtype=np.float32)
    train_data_probs = np.ndarray(shape=(args['train_size']), dtype=np.float32)
    for x in range(args['train_size']):
        p = np.random.uniform(0,1)
        i = np.searchsorted(cum_probs, p)
        train_data[x][:] = all_outcomes[i]
        train_data_probs[x] = prob_of_outcomes[i]
    
    
    
    valid_data = np.ndarray(shape=(args['validation_size'], graph_size), dtype=np.float32)
    valid_data_probs = np.ndarray(shape=(args['validation_size']), dtype=np.float32)
    for x in range(args['validation_size']):
        p = np.random.uniform(0,1)
        i = np.searchsorted(cum_probs, p)
        valid_data[x][:] = all_outcomes[i]
        valid_data_probs[x] = prob_of_outcomes[i]
        
    
    if args['full_test'] == True:
        test_data = all_outcomes.copy()
        test_data_probs = prob_of_outcomes.copy()
    else:
        test_data = np.ndarray(shape=(args['test_size'], graph_size), dtype=np.float32)
        test_data_probs = np.ndarray(shape=(args['test_size']), dtype=np.float32)
        for x in range(args['test_size']):
            
            if np.mod(x,100) == 0:
                print('         ' + str(x))
            
            p = np.random.uniform(0,1)
            i = np.searchsorted(cum_probs, p)
            test_data[x][:] = all_outcomes[i]
        test_data_probs[x] = prob_of_outcomes[i]
    #test_data = all_outcomes[prob_of_outcomes > 0][:]
    #test_data_probs = prob_of_outcomes[prob_of_outcomes > 0]
    
    return {'train_data' : train_data, 
            'train_data_probs' : train_data_probs,  
            'valid_data' : valid_data,
            'valid_data_probs' : valid_data_probs,
            'test_data' : test_data,
            'test_data_probs' : test_data_probs}

def _get_mnist_data(args):
    return None


def get_data(args):
    if config.data_name == 'grid':
        args['parameters_file'] = 'dataset_parameters/grid' + str(config.height) + 'by' + str(config.width) + '_parameters.npz'
        args['full_test'] = True
        return _get_Ising_data(args)
    elif config.data_name == 'Boltzmann':
        args['parameters_file'] = 'dataset_parameters/Boltzman_' + str(config.n_boltzmann) + ',' + str(config.m_boltzmann) + '_parameters.npz'
        args['full_test'] = True
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
