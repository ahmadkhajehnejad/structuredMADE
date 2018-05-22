import numpy as np

def _get_Ising_data_params(w_v, w_e):
    
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
    
    return [all_outcomes, prob_of_outcomes, cum_probs]


def gen_params(args):
    if args['data_name'] == 'grid':
        height = args['height']
        width = args['width']
        graph_size = height*width
        
        w_v = np.random.sample(graph_size)
        w_e = np.zeros([graph_size, graph_size])
        for r in range(0, height):
            for c in range(0, width):
                jj = r*width + c
                if c > 0:
                    param = np.random.sample(1)
                    w_e[jj-1][jj] = w_e[jj][jj-1] = param
                if r > 0:
                    param = np.random.sample(1)
                    w_e[jj-width][jj] = w_e[jj][jj-width] = param
                    
        all_outcomes, prob_of_outcomes, cum_probs = _get_Ising_data_params(w_v, w_e)            
        np.savez('dataset_parameters/grid' + str(height) + 'by' + str(width) + '_parameters.npz',
                 edge_parameters = w_e,
                 vertex_parameters = w_v,
                 all_outcomes = all_outcomes,
                 prob_of_outcomes = prob_of_outcomes,
                 cum_probs = cum_probs,
                 graph_size = len(w_v)
                 )
    elif args['data_name'] == 'Boltzmann':
        n = args['n']
        m = args['m']
        
        w_v = np.random.sample(n+m)
        w_e = np.zeros([n+m,n+m])
        for i in range(n):
            for j in range(m):
                param = np.random.sample(1)
                w_e[i,n+j] = w_e[n+j,i] = param
                
        all_outcomes, prob_of_outcomes, cum_probs = _get_Ising_data_params(w_v, w_e)
        np.savez('dataset_parameters/Boltzman_' + str(n) + ',' + str(m) + '_parameters.npz',
                 edge_parameters = w_e,
                 vertex_parameters = w_v,
                 all_outcomes = all_outcomes,
                 prob_of_outcomes = prob_of_outcomes,
                 cum_probs = cum_probs,
                 graph_size = len(w_v)
                 )
#gen_params({'data_name' : 'grid', 'height' : 4, 'width' : 4})
#gen_params({'data_name' : 'Boltzmann', 'n' : 10, 'm' : 10})
