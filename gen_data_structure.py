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
        
gen_structure({'data_name': 'k_sparse', 'n': 20, 'sparsity_degree': 3})