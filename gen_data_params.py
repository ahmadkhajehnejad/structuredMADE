import numpy as np

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
                    
        np.savez('dataset_parameters/grid' + str(height) + 'by' + str(width) + '_parameters.npz',
                 edge_parameters = w_e,
                 vertex_parameters = w_v
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
        np.savez('dataset_parameters/Boltzman_' + str(n) + ',' + str(m) + '_parameters.npz',
                 edge_parameters = w_e,
                 vertex_parameters = w_v
                 )        
#gen_params({'data_name' : 'grid', 'height' : 4, 'width' : 4})
#gen_params({'data_name' : 'Boltzmann', 'n' : 10, 'm' : 10})
