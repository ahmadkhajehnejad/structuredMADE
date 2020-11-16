import config
import numpy as np
import sys
if config.use_multiprocessing:
    from multiprocessing import Process, Queue


def _spread(current_node, root_node, visited, adj, pi):
        visited[current_node]=True
        if pi[current_node] < pi[root_node]:
            return
        for nei in range(adj.shape[0]):
            if adj[current_node,nei] == 1 and not visited[nei]:
                _spread(nei, root_node, visited, adj, pi)


def thread_func(shared_space,n,adj,pi,index):
    visited = np.zeros([n],dtype=bool)
    _spread(current_node = index, root_node = index, visited = visited, adj=adj, pi=pi)
    visited[pi >= pi[index]] = False
    if config.use_multiprocessing:
        shared_space.put( [index, np.where(visited)[0]] )
    else:
        return np.where(visited)[0]
    #print(index,' finished')
    #sys.stdout.flush()


def _make_Q(adj, pi):
    n = adj.shape[0]
    if config.use_multiprocessing:
        shared_space = Queue()
        thr = [None] * n
        Q = [None] * n
        for i in range(n):
            #thr[i] = threading.Thread(target=thread_func, args=[Q,adj,pi,i])
            thr[i] = Process(target=thread_func, args=[shared_space,n,adj,pi,i])
            thr[i].start()
        for i in range(n):
            res = shared_space.get()
            Q[res[0]] = res[1]
        for i in range(n):
            thr[i].join()
            thr[i].terminate()
        shared_space.close()
    else:
        Q = [None] * n
        for i in range(n):
            Q[i] = thread_func(None,n,adj,pi,i)
    return Q


def _detect_subsets(labels_1, labels_2):
    tmp1 = np.zeros([len(labels_1), config.graph_size])
    tmp2 = np.zeros([len(labels_2), config.graph_size])

    for k in range(len(labels_1)):
        tmp1[k,labels_1[k]] = 1
    for k in range(len(labels_2)):
        tmp2[k,labels_2[k]] = 1

    tmp3 = np.matmul(tmp1, (1-tmp2).T)

    return np.array((tmp3 == 0), dtype=np.float32)
