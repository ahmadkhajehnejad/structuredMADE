import bfs_orders
import config
from dataset import get_data_structure
from utils import grid_orders
from utils.masking_utils import _make_Q, _detect_subsets
import numpy as np

class MADE_base:
    def __init__(self, all_masks=None, all_pi=None):
        self.masking_method = config.algorithm
        if (self.masking_method in ['Q_restricted', 'random_Q_restricted' , 'ensemble_Q_restricted_and_orig', 'min_related']) or \
                (config.random_dimensions_order in ['bfs']) or \
                (str(config.random_dimensions_order).endswith('bfs-random')):
            parameters = get_data_structure()
            self.adjacency_matrix = parameters['adjacency_matrix']
        if (all_masks is None) or (all_pi is None):
            self.all_masks, self.all_pi = self.generate_all_masks()
        else:
            self.all_masks = all_masks.copy()
            self.all_pi = all_pi.copy()



    def generate_all_masks(self):
        all_masks = []
        all_pi = []

        for i_m in range(0,config.num_of_all_masks):

            print('mask', i_m)

            if config.random_dimensions_order == False:
                pi = np.arange(config.graph_size)
            elif config.random_dimensions_order == True:
                pi = np.random.permutation(config.graph_size)
            elif config.random_dimensions_order == 'grid':
                pi = grid_orders.get_random_order(config.width, config.height)
            elif config.random_dimensions_order == 'bfs':
                pi = bfs_orders.get_random_order(self.adjacency_matrix)
            elif config.random_dimensions_order.startswith('grid_partial_random'):
                num_parts = int(config.random_dimensions_order[20:])
                pi = grid_orders.get_partially_random_order(config.width, config.height, num_parts, False)
            elif config.random_dimensions_order.startswith('fixed_partial_random'):
                num_parts = int(config.random_dimensions_order[21:])
                pi = grid_orders.get_partially_random_order(config.width, config.height, num_parts, True)
            elif config.random_dimensions_order.endswith('grid-random'):
                if i_m < int(config.random_dimensions_order[:-11]):
                    pi = grid_orders.get_random_order(config.width, config.height)
                else:
                    pi = np.random.permutation(config.graph_size)
            elif config.random_dimensions_order.endswith('bfs-random'):
                if i_m < int(config.random_dimensions_order[:-10]):
                    pi = bfs_orders.get_random_order(self.adjacency_matrix)
                else:
                    pi = np.random.permutation(config.graph_size)
            elif config.random_dimensions_order == 'grid_from_center':
                pi = grid_orders.get_random_order_from_center(config.width, config.height)
            else:
                raise Exception('Error')
            all_pi.append(pi)

            ################### pi = np.arange(config.graph_size)  ## We run the permutation on the data, not the labels of the first and last layer
            if self.masking_method == 'Q_restricted':
                all_masks.append(self._Q_restricted_mask(pi))
            elif self.masking_method == 'random_Q_restricted':
                all_masks.append(self._Q_restricted_mask(pi,random_Q=True))
            elif self.masking_method == 'ensemble_Q_restricted_and_orig':
                if i_m < config.num_of_all_masks // 2:
                    all_masks.append(self._Q_restricted_mask(pi))
                else:
                    all_masks.append(self._normal_mask('orig',pi))
            else:
                all_masks.append(self._normal_mask(self.masking_method,pi))

        swapped_all_masks = []
        for i in range(config.num_of_hlayer+1):
            swapped_masks = []
            for j in range(config.num_of_all_masks):
                swapped_masks.append(all_masks[j][i])
                # swapped_masks.append( reduce(lambda a, b: 2 * a + b, all_masks[j][i].T) )
            swapped_all_masks.append(swapped_masks)


        #all_masks = [[x*1.0 for x in y] for y in all_masks]

        return swapped_all_masks, all_pi

    def _Q_restricted_mask(self, pi, random_Q=False):
        ### We can change it
        mu = [(i+1)/(config.num_of_hlayer+1) for i in range(config.num_of_hlayer)]

        Q = _make_Q(self.adjacency_matrix, pi)

        if random_Q:
            for i in range(config.graph_size):
                if pi[i] > 0:
                    precedings = np.where(pi < pi[i])[0]
                    Q[i] = np.random.choice(precedings, len(Q[i]), replace=False)


        labels = []
        for i in range(config.num_of_hlayer):
            labels.append([None] * config.hlayer_size)
        labels.insert(0,[np.array([i]) for i in range(config.graph_size)])
        labels.append( [Q[i] for i in range(config.graph_size)])

        for i in range(1,config.num_of_hlayer+1):
            corresp_Q_inds = np.random.randint(0,config.graph_size,config.hlayer_size)
            rnd_prm = np.random.permutation(config.hlayer_size)
            corresp_Q_inds[ rnd_prm[:config.graph_size] ] = np.arange(0,config.graph_size)


            tmp0 = np.zeros([len(labels[i-1]), config.graph_size])
            for k in range(len(labels[i-1])):
                tmp0[k,labels[i-1][k]] = 1


            for j in range(config.hlayer_size):

                corresp_Q = Q[corresp_Q_inds[j]]
                while corresp_Q.size == 0:
                    corresp_Q = Q[np.random.randint(0,config.graph_size)]

                tmp_Q = np.zeros([config.graph_size,1])
                tmp_Q[corresp_Q,0] = 1

                tmp = np.where(np.matmul(tmp0, 1-tmp_Q) == 0)[0]

                '''
                tmp_ = [k for k in range(len(labels[i-1])) if np.intersect1d(labels[i-1][k], corresp_Q).size == np.unique(labels[i-1][k]).size]
                if not(np.all(tmp == tmp_)):
                    while (True):
                        print('tmp != tmp' )
                '''

                pr = np.random.choice(tmp)


                labels[i][j] = (labels[i-1][pr]).copy()

                rnd = np.random.uniform(0.,1.,corresp_Q.size)

                labels[i][j] = np.union1d(labels[i][j], corresp_Q[rnd < mu[i-1]])


        #cnt_masks = 0
        masks = []
        for i in range(1,len(labels)):

            #print('layer # ', i)

            mask = _detect_subsets(labels[i - 1], labels[i])

            '''
            mask_2 = np.zeros([len(labels[i-1]), len(labels[i])], dtype=np.float32)
            for k in range(len(labels[i-1])):
                for j in range(len(labels[i])):
                    mask_2[k][j] = (np.intersect1d(labels[i-1][k], labels[i][j]).size == np.unique(labels[i-1][k]).size)
                    #if mask[k][j]:
                    #    cnt_masks += 1
            print('mask == maks_2 :  ', np.all(mask == mask_2))
            '''

            masks.append(mask)


        ## Backward Pass
        if config.Q_restricted_2_pass:
            for i in reversed(range(1,len(labels)-1)):
                is_subset = _detect_subsets(labels[i - 1], labels[i + 1])


                zero_out_edge_nodes = np.where(np.sum(masks[i], axis=1) == 0)[0]
                for j in zero_out_edge_nodes:
                    t = np.random.randint(0,len(labels[i+1]))
                    k = np.random.choice( np.where(is_subset[:,t].reshape([-1]) > 0)[0] )

                    labels[i][j] = (labels[i-1][k]).copy()
                    rnd = np.random.uniform(0.,1.,labels[i+1][t].size)
                    labels[i][j] = np.union1d(labels[i][j], labels[i+1][t][rnd < mu[i-1]])

                masks[i] = _detect_subsets(labels[i], labels[i + 1])
                masks[i-1] = _detect_subsets(labels[i - 1], labels[i])


        if config.direct_links != False:
            tmp_mask = np.zeros([config.graph_size, config.graph_size], dtype=np.float32)
            for j in range(config.graph_size):
                if config.direct_links == 'Full':
                    tmp_mask[pi < pi[j], j] = 1.0
                elif config.direct_links == True:
                    tmp_mask[Q[j], j] = 1.0
                else:
                    raise Exception('Error' + str(config.direct_links))
            masks[-1] = np.concatenate([masks[-1], tmp_mask], axis=0)


        #print('-- ' + str(cnt_masks))
        return masks

    def _normal_mask(self, masking_method, pi):

        if masking_method == 'min_related':
            Q = _make_Q(self.adjacency_matrix, pi)
            min_related_pi = np.zeros([config.graph_size])
            for i in range(len(Q)):
                if len(Q[i]) > 0:
                    min_related_pi[pi[i]] = pi[Q[i]].min()
                else:
                    min_related_pi[pi[i]] = pi[i]
            #min_related_pi[pi] = np.array([pi[q].min() if len(q)>0 else  for q in Q])
            for i in reversed(range(config.graph_size-1)):
                min_related_pi[i] = min(min_related_pi[i], min_related_pi[i+1])
            #related_size = pi - min_related_pi
        elif masking_method != 'orig':
            raise Exception('Error')

        labels = [pi] # np.zeros([config.num_of_hlayer, config.hlayer_size], dtype=int)
        min_label = 0
        for _ in range(config.num_of_hlayer):
            labels.append(np.random.randint(min_label, config.graph_size, (config.hlayer_size)))
            min_label = np.amin(labels[-1])
        labels.append(pi)

        masks = []

        #hidden layers mask
        for i in range(config.num_of_hlayer):
            mask = np.zeros([labels[i].size, labels[i+1].size], dtype=np.float32)
            for j in range(0, labels[i+1].size):
                for k in range(0, labels[i].size):
                    if (masking_method == 'orig'):
                        if (labels[i+1][j] >= labels[i][k]):
                            mask[k][j] = 1.0
                    elif masking_method == 'min_related':
                        if ((labels[i+1][j] >= labels[i][k]) and (min_related_pi[labels[i+1][j]] <= labels[i][k] )):
                            mask[k][j] = 1.0
                    else:
                        raise Exception("wrong masking method " + masking_method)

            masks.append(mask)

        #last layer mask
        mask = np.zeros([labels[-2].size, labels[-1].size], dtype=np.float32)
        for j in range(0, labels[-1].size):
            for k in range(0, labels[-2].size):
                if (masking_method == 'orig'):
                    if (labels[-1][j] > labels[-2][k]):
                        mask[k][j] = 1.0
                elif (masking_method == 'min_related'):
                    if ((labels[-1][j] > labels[-2][k]) and (min_related_pi[labels[-1][j]] <= labels[-2][k])):
                        mask[k][j] = 1.0
                else:
                    raise Exception("wrong masking method " + masking_method)

        if config.direct_links != False:
            tmp_mask = np.zeros([config.graph_size, config.graph_size], dtype=np.float32)
            for j in range(0,config.graph_size):
                if masking_method == 'orig':
                    if (config.direct_links == True) or (config.direct_links == 'Full'):
                        tmp_mask[pi < pi[j], j] = 1.0
                    else:
                        raise Exception('Error' + str(config.direct_links))
                elif masking_method == 'min_related':
                    if config.direct_links == 'Full':
                        tmp_mask[pi < pi[j], j] = 1.0
                    elif config.direct_links == True:
                        ind = (pi < pi[j]) & (pi >= (min_related_pi[pi[j]]))
                        if np.any(ind):
                            tmp_mask[ind, j] = 1.0
                    else:
                        raise Exception('Error' + str(config.direct_links))
            mask = np.concatenate([mask, tmp_mask],axis=0)

        masks.append(mask)
        return masks
