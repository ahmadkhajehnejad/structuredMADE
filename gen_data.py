import numpy as np
from keras.datasets import mnist

def _gen_Ising_data(args):
    
    with np.load(args['parameters_file']) as parameters:
        all_outcomes = parameters['all_outcomes']
        prob_of_outcomes = parameters['prob_of_outcomes']
        cum_probs = parameters['cum_probs']

    p = np.random.uniform(0,1,args['train_size'])
    ind = np.searchsorted(cum_probs, p)
    train_data = all_outcomes[ind,:]
    train_data_probs = prob_of_outcomes[ind]           

    p = np.random.uniform(0,1,args['test_size'])
    ind = np.searchsorted(cum_probs, p)
    test_data = all_outcomes[ind,:]
    test_data_probs = prob_of_outcomes[ind]
            
    return {'train_data' : train_data,
            'train_data_probs' : train_data_probs,
            'test_data' : test_data,
            'test_data_probs' : test_data_probs,
            'all_outcomes': all_outcomes,
            'prob_of_outcomes': prob_of_outcomes
            }

def _gen_mnist_data(args):
    #bmnist1 = np.load('datasets/binary_mnist_1.npz')
    special_digit = args['digit']
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    trd = np.rint(x_train[y_train == special_digit]/255.0)
    np.random.shuffle(trd)
    #train_data = np.reshape(trd[:,7:21,7:21], (trd.shape[0], 14*14))
    train_data = np.reshape(trd, (trd.shape[0], -1))
    ted = np.rint(x_test[y_test == special_digit]/255.0)
    np.random.shuffle(ted)
    #test_data = np.reshape(ted[:,7:21,7:21], (ted.shape[0], 14*14))
    test_data = np.reshape(ted, (ted.shape[0], -1))
    return {'train_data' : train_data,
            'test_data' : test_data}
    #img = Image.fromarray(trd[154]*255)
    #img.show()


def gen_data(args):
    if args['data_name'] == 'grid':
        args['parameters_file'] = 'dataset_parameters/grid' + str(args['height']) + 'by' + str(args['width']) + '_parameters.npz'
        dt = _gen_Ising_data(args)
        dest_file = 'datasets/grid' + str(args['height']) + 'by' + str(args['width']) + '.npz'
    elif args['data_name'] == 'Boltzmann':
        args['parameters_file'] = 'dataset_parameters/Boltzman_' + str(args['n']) + '&' + str(args['m']) + '_parameters.npz'
        dt = _gen_Ising_data(args)
        dest_file = 'datasets/Boltzman_' + str(args['n']) + '&' + str(args['m']) + '.npz'
    elif args['data_name'] == 'k_sparse':
        args['parameters_file'] = 'dataset_parameters/k_sparse_' + str(args['n']) + '_' + str(args['sparsity_degree']) + '_parameters.npz'
        dt = _gen_Ising_data(args)
        dest_file = 'datasets/k_sparse_' + str(args['n']) + '_' + str(args['sparsity_degree']) + '.npz'
    elif args['data_name'] == 'mnist':
        dt = _gen_mnist_data(args)
        dt['all_outcomes'] = None
        dt['prob_of_outcomes'] = None
        dt['train_data_probs'] = None
        dt['test_data_probs'] = None
        dest_file = 'datasets/binary_mnist_' + str(args['digit']) + '.npz'
    elif args['data_name'] == 'rcv1':
        rcv1 = np.load('datasets/rcv1_orig.npz')
        tr = np.concatenate( [rcv1['train_data'], rcv1['valid_data']] ).copy()
        te = rcv1['test_data'].copy()
        np.random.shuffle(tr)
        np.random.shuffle(te)
        dt = {'train_data': tr, 'test_data': te}
        dt['all_outcomes'] = None
        dt['prob_of_outcomes'] = None
        dt['train_data_probs'] = None
        dt['test_data_probs'] = None
        dest_file = 'datasets/rcv1.npz'
    
        
        
    np.savez(dest_file,
             train_data = dt['train_data'],
             train_data_probs = dt['train_data_probs'],
             test_data = dt['test_data'],
             test_data_probs = dt['test_data_probs'],
             all_outcomes = dt['all_outcomes'],
             prob_of_outcomes = dt['prob_of_outcomes']
             )

#gen_data({'data_name' : 'grid', 'height' : 4, 'width' : 4, 'train_size' : 20000, 'test_size' : 100000})
#gen_data({'data_name' : 'Boltzmann', 'n' : 10, 'm' : 10, 'train_size' : 20000, 'test_size' : 100000})
#gen_data({'data_name' : 'k_sparse', 'n' : 20, 'sparsity_degree' : 3, 'train_size' : 20000, 'test_size' : 100000})
#gen_data({'data_name' : 'rcv1'})
'''
for d in range(10):
    print(d)
    gen_data({'data_name' : 'mnist', 'digit' : d})
'''