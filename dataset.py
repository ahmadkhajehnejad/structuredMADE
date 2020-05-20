import numpy as np
import config
import h5py
from PIL import Image

def _get_data_from_file(args):
    
    with np.load(args['data_file']) as parameters:
        
        all_outcomes = parameters['all_outcomes']
        prob_of_outcomes = parameters['prob_of_outcomes']
        
        data = parameters['train_data']
        data_probs = parameters['train_data_probs']

        if config.random_data:
            rnd_prm = np.random.permutation(len(data))
            data = data[rnd_prm]
            #if not(data_probs == None):
            #if (data_probs is not None):
            if len(data_probs.shape) > 0:
                data_probs = data_probs[rnd_prm]

        train_data = data[0:args['train_size']]
        valid_data = data[args['train_size']:(args['train_size']+args['valid_size'])]
        
        
        #if not(data_probs == None):
        #if data_probs is not None:
        if len(data_probs.shape) > 0:
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

            if config.random_data:
                rnd_prm = np.random.permutation(len(data))
                data = data[rnd_prm]

                #if not(data_probs == None):
                #if data_probs is not None:
                if len(data_probs.shape) > 0:
                    data_probs = data_probs[rnd_prm]
            test_data = data[0:args['test_size']]

            #if not(data_probs == None):
            #if data_probs is not None:
            if len(data_probs.shape) > 0:
                test_data_probs = data_probs[0:args['test_size']]
            else:
                test_data_probs = None

    if config.use_uniform_noise_for_pmf:
        [train_data, valid_data, test_data] = [
            np.tile(x, [config.num_noisy_samples_per_sample, 1]) +
            np.random.rand(x.shape[0] * config.num_noisy_samples_per_sample * x.shape[1]).reshape(
                [x.shape[0] * config.num_noisy_samples_per_sample, x.shape[1]]
            )
            for x in [train_data, valid_data, test_data]
        ]

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
    elif args['data_name'].startswith('mnist'):
        if args['digit'] == 'All':
            tr = args['train_size']
            va = args['valid_size']
            te = args['test_size']
            args['train_size'] = args['train_size'] // 10
            args['valid_size'] = args['valid_size'] // 10
            args['test_size'] = args['test_size'] // 10
            
            args['data_file'] = 'datasets/mnist_'+ str(0) + '.npz'
            res = _get_data_from_file(args)
            for d in range(1,10):
                args['data_file'] = 'datasets/mnist_'+ str(d) + '.npz'
                tmp = _get_data_from_file(args)
                res['train_data'] = np.concatenate([res['train_data'], tmp['train_data']], axis=0)
                res['valid_data'] = np.concatenate([res['valid_data'], tmp['valid_data']], axis=0)
                res['test_data'] = np.concatenate([res['test_data'], tmp['test_data']], axis=0)
            np.random.shuffle(res['train_data'])
            np.random.shuffle(res['valid_data'])
            np.random.shuffle(res['test_data'])
            
            args['train_size'] = tr
            args['valid_size'] = va
            args['test_size'] = te
            
            return res
        else:
            raise Exception("ERROR: mnist should not be run for just one digit")
            #args['data_file'] = 'datasets/binary_mnist_'+ str(args['digit']) + '.npz'
            #return _get_data_from_file(args)
    elif args['data_name'].startswith('binarized_mnist'):
        if args['digit'] == 'All':
            tr = args['train_size']
            va = args['valid_size']
            te = args['test_size']
            args['train_size'] = args['train_size'] // 10
            args['valid_size'] = args['valid_size'] // 10
            args['test_size'] = args['test_size'] // 10

            args['data_file'] = 'datasets/binary_mnist_' + str(0) + '.npz'
            res = _get_data_from_file(args)
            for d in range(1, 10):
                args['data_file'] = 'datasets/binary_mnist_' + str(d) + '.npz'
                tmp = _get_data_from_file(args)
                res['train_data'] = np.concatenate([res['train_data'], tmp['train_data']], axis=0)
                res['valid_data'] = np.concatenate([res['valid_data'], tmp['valid_data']], axis=0)
                res['test_data'] = np.concatenate([res['test_data'], tmp['test_data']], axis=0)
            np.random.shuffle(res['train_data'])
            np.random.shuffle(res['valid_data'])
            np.random.shuffle(res['test_data'])

            args['train_size'] = tr
            args['valid_size'] = va
            args['test_size'] = te

            return res
        else:
            raise Exception("ERROR: mnist should not be run for just one digit")
            # args['data_file'] = 'datasets/binary_mnist_'+ str(args['digit']) + '.npz'
            # return _get_data_from_file(args)
    elif args['data_name'].startswith('ocr'):
        tr = args['train_size']
        va = args['valid_size']
        te = args['test_size']

        args['train_size'] = tr // 20
        args['valid_size'] = va // 20
        args['test_size'] = te // 20
        args['data_file'] = 'datasets/ocr_' + str(config.ocr_characters[0]) + '.npz'
        res = _get_data_from_file(args)
        for d in range(1, 20):
            args['train_size'] = tr // 20
            args['valid_size'] = va // 20
            args['test_size'] = te // 20
            args['data_file'] = 'datasets/ocr_' + str(config.ocr_characters[d]) + '.npz'

            tmp = _get_data_from_file(args)
            res['train_data'] = np.concatenate([res['train_data'], tmp['train_data']], axis=0)
            res['valid_data'] = np.concatenate([res['valid_data'], tmp['valid_data']], axis=0)
            res['test_data'] = np.concatenate([res['test_data'], tmp['test_data']], axis=0)

        np.random.shuffle(res['train_data'])
        np.random.shuffle(res['valid_data'])
        np.random.shuffle(res['test_data'])


        #################### in baayad avaz beshe!!!
        args['train_size'] = tr
        args['valid_size'] = va
        args['test_size'] = te

        return res
    elif args['data_name'].startswith('cifar10'):
        args['data_file'] = 'datasets/cifar10.npz'
        res = _get_data_from_file(args)
        np.random.shuffle(res['train_data'])
        np.random.shuffle(res['valid_data'])
        np.random.shuffle(res['test_data'])
        return res
    elif args['data_name'].startswith('celebA'):
        tr = args['train_size']
        va = args['valid_size']
        te = args['test_size']

        celebA_dataset_path = './datasets/celebA/'

        num_all_images = 202599
        new_size = config.new_size

        if config.random_data:
            dt_ind = np.load(celebA_dataset_path + 'rndprm.npy')
        else:
            dt_ind = np.random.permutation(num_all_images)

        train_data = []
        for i in range(tr):
            filename = str(dt_ind[i] + 1)
            while len(filename) < 6:
                filename = '0' + filename
            img = np.asarray(Image.open(celebA_dataset_path + filename + '.jpg').resize(new_size)) / 255
            train_data.append(img.reshape([-1]))
        train_data = np.array(train_data)

        valid_data = []
        for i in range(va):
            filename = str(dt_ind[tr + i] + 1)
            while len(filename) < 6:
                filename = '0' + filename
            img = np.asarray(Image.open(celebA_dataset_path + filename + '.jpg').resize(new_size)) / 255
            valid_data.append(img.reshape([-1]))
        valid_data = np.array(valid_data)

        test_data = []
        for i in range(te):
            filename = str(dt_ind[tr + va + i] + 1)
            while len(filename) < 6:
                filename = '0' + filename
            img = np.asarray(Image.open(celebA_dataset_path + filename + '.jpg').resize(new_size)) / 255
            test_data.append(img.reshape([-1]))
        test_data = np.array(test_data)

        np.random.shuffle(train_data)
        np.random.shuffle(valid_data)
        np.random.shuffle(test_data)

        return {'train_data': train_data,
                'train_data_probs': None,
                'valid_data': valid_data,
                'valid_data_probs': None,
                'test_data': test_data,
                'test_data_probs': None}

    elif args['data_name'] == 'k_sparse':
        args['data_file'] = 'datasets/k_sparse_' + str(args['n']) + '_' + str(args['sparsity_degree']) + '.npz'
        return _get_data_from_file(args)
    elif args['data_name'] == 'rcv1':
        args['data_file'] = 'datasets/rcv1.npz'
        return _get_data_from_file(args)
    elif args['data_name'] == 'BayesNet':
        args['data_file'] = 'datasets/BayesNet_' + str(args['n']) + '_' + str(args['par_num']) + '.npz'
        return _get_data_from_file(args)
    else:
        return None

def make_hdf5(args, filename):
    data = get_data(args)
    with h5py.File(filename, 'w') as f:
        tr = f.create_group("train")
        tr.create_dataset("data", data = np.array(data['train_data'], dtype=np.float32))
        va = f.create_group("validation")
        va.create_dataset("data", data = np.array(data['valid_data'], dtype=np.float32))
        te = f.create_group("test")
        te.create_dataset("data", data = np.array(data['test_data'], dtype=np.float32))

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

    elif config.data_name.startswith('binarized_mnist') or config.data_name.startswith(
        'mnist') or config.data_name.startswith('ocr') or config.data_name.startswith('celebA')\
            or config.data_name.startswith('cifar10'):
        graph_size = config.graph_size
        if config.data_name.startswith('binarized_mnistdps') or config.data_name.startswith('mnistdps') \
            or config.data_name.startswith('ocrdps') or config.data_name.startswith('celebAdps')\
                or config.data_name.startswith('cifar10dps'):
            if config.data_name.startswith('binarized_mnistdps'):
                dp = int(config.data_name[18:])
            elif config.data_name.startswith('mnistdps'):
                dp = int(config.data_name[8:])
            elif config.data_name.startswith('cifar10dps'):
                dp = 3 * int(config.data_name[10:]) + 2
            elif config.data_name.startswith('celebAdps'):
                dp = int(config.data_name[9:])
            else:
                dp = int(config.data_name[6:])
            adj = np.zeros([graph_size, graph_size])
            for r in range(0, config.height):
                for c in range(0, config.width):
                    jj = r*config.width + c
                    
                    for t_r in range(-dp, dp+1):
                        for t_c in range(-dp, dp+1):
                            if (not((t_r == 0) and (t_c == 0))) and (0 <= r+t_r < config.height) and (0 <= c+t_c < config.width):
                                zz = (r+t_r)*config.width + (c+t_c)
                                adj[jj,zz] = adj[zz,jj] = 1
            parameters['adjacency_matrix'] = adj
        elif config.data_name.startswith('binarized_mnistdp') or config.data_name.startswith('mnistdp') \
            or config.data_name.startswith('ocrdp') or config.data_name.startswith('celebAdp'):
            if config.data_name.startswith('binarized_mnistdp'):
                dp = int(config.data_name[17:])
            elif config.data_name.startswith('mnistdp'):
                dp = int(config.data_name[7:])
            elif config.data_name.startswith('cifar10dp'):
                dp = 3 * int(config.data_name[9:]) + 2
            elif config.data_name.startswith('celebAdp'):
                dp = int(config.data_name[8:])
            else:
                dp = int(config.data_name[5:])
            adj = np.zeros([graph_size, graph_size])
            for r in range(0, config.height):
                for c in range(0, config.width):
                    jj = r*config.width + c
                    
                    for t_r in range(-dp, dp+1):
                        for t_c in range(-dp, dp+1):
                            if (0 < np.abs(t_r) + np.abs(t_c) <= dp) and (0 <= r+t_r < config.height) and (0 <= c+t_c < config.width):
                                zz = (r+t_r)*config.width + (c+t_c)
                                adj[jj,zz] = adj[zz,jj] = 1
            parameters['adjacency_matrix'] = adj
        else:
            raise('Unknown dataset: ' + config.data_name)
        
    elif config.data_name == 'Boltzmann':
        n = config.n_boltzmann
        m = config.m_boltzmann
        graph_size = n + m
        adj = np.zeros([graph_size, graph_size])
        for i in range(n):
            for j in range(m):
                adj[i,n+j] = adj[n+j,i] = 1
        parameters['adjacency_matrix'] = adj
        
    elif config.data_name == 'k_sparse':
        with np.load('dataset_structures/k_sparse_' + str(config.n_of_k_sparse) + '_' + str(config.sparsity_degree) + '_structure.npz') as params:
            parameters['adjacency_matrix'] = params['adjacency_matrix']
    elif config.data_name == 'rcv1':
        with np.load('dataset_structures/rcv1_structure.npz') as params:
            parameters['adjacency_matrix'] = params['adjacency_matrix']
    elif config.data_name == 'BayesNet':
        with np.load('dataset_structures/BayesNet_' + str(config.n_of_BayesNet) + '_' + str(config.par_num_of_BayesNet) + '_structure.npz') as params:
            parameters['adjacency_matrix'] = params['adjacency_matrix']        
    return parameters

#data = get_data({'data_name' : 'ocr', 'train_size' : 200, 'valid_size' : 50, 'test_size': 5000})
#print(data['train_data'].shape[0])
#print(data['valid_data'].shape[0])
#print(data['test_data'].shape[0])
