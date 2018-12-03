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
    if args['data_name'] == 'BayesNet':
        n = args['n']
        par_num = args['par_num']
        adjacency_matrix = np.zeros([n,n])
        par = [None] * n
        par[0] = np.array([])
        for i in range(1,n):
            par[i] = np.random.choice(list(range(i)),np.minimum(par_num, i),replace=False)
            adjacency_matrix[i,par[i]] = 1
            adjacency_matrix[par[i],i] = 1
            for j in par[i]:
                for t in par[i]:
                    if j != t:
                        adjacency_matrix[j,t] = 1
                        adjacency_matrix[t,j] = 1
        np.savez('dataset_structures/BayesNet_' + str(n) + '_' + str(par_num) + '_structure.npz',
                 adjacency_matrix = adjacency_matrix,
                 par = par
                )

gen_structure({'data_name': 'BayesNet', 'n': 100, 'par_num': 5})
#gen_structure({'data_name': 'k_sparse', 'n': 100, 'sparsity_degree': 5})
#gen_structure({'data_name': 'rcv1'})