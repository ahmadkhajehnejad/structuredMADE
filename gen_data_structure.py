import numpy as np

def gen_structure(args):
    if args['data_name'] == 'k_sparse':
        n = args['n']
        sparsity_degree = args['sparsity_degree']
        adjacency_matrix = np.zeros([n,n])
        for i in range(1,n):
            ind = np.random.choice(list(range(i)),np.minimum(sparsity_degree, i),replace=False)
            adjacency_matrix[i,ind] = 1
            adjacency_matrix[ind,i] = 1
        np.savez('dataset_structures/k_sparse_' + str(n) + '_' + str(sparsity_degree) + '_structure.npz',
                 adjacency_matrix = adjacency_matrix
                )
    if args['data_name'] == 'rcv1':
        with np.load('datasets/rcv1_orig.npz') as rcv1:
            cov = np.cov( np.concatenate([ rcv1['train_data'], rcv1['valid_data']]).T )
            adjacency_matrix = np.array(np.abs(cov) > 0.02, dtype=int)
            np.savez('dataset_structures/rcv1_structure.npz', adjacency_matrix = adjacency_matrix)
        
#gen_structure({'data_name': 'k_sparse', 'n': 20, 'sparsity_degree': 3})
#gen_structure({'data_name': 'rcv1'})