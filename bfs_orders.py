import numpy as np

def get_random_order(adjacency_matrix):
    graph_size = adjacency_matrix.shape[0]
    v = np.random.randint(graph_size)
    
    mark = np.zeros([graph_size])
    queue = np.zeros([graph_size], dtype=int)
    pi = np.zeros([graph_size], dtype=np.int)
    mark[v] = 1
    head = tail = 0
    queue[0] = v
    pi[v] = 0
    
    while head <= tail:
        nei = np.where(adjacency_matrix[queue[head],:] == 1)[0]
        for u in nei:
            if mark[u] == 0:
                mark[u] = 1
                tail += 1
                queue[tail] = u
                pi[u] = tail
        head += 1
        
    return pi