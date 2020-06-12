import numpy as np
from keras.models import Model
from keras.layers import Input, Concatenate, Reshape, Flatten, Add, Average, Multiply, Lambda
import config
import keras
from keras import backend as K
from made_utils import MaskedDenseLayer, MyEarlyStopping, MaskedConvLayer
from dataset import get_data_structure
#from keras import optimizers
import keras
import grid_orders
import bfs_orders
#import threading
if config.use_multiprocessing:
    from multiprocessing import Process, Queue
#import sys
from sklearn.linear_model import LogisticRegression
from functools import reduce
from scipy.special import logsumexp

def _spread(current_node, root_node, visited, adj, pi):
        visited[current_node]=True
        if pi[current_node] < pi[root_node]:
            return
        for nei in range(adj.shape[0]):
            if adj[current_node,nei] == 1 and not visited[nei]:
                _spread(nei, root_node, visited, adj, pi)
                

#def thread_func(Q,adj,pi,index):
    #n = len(Q)
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
    
    

class MADE:
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

        #print('avg reaching sizes: ', np.mean(self.reaching_dimesions_num()))

        if config.learn_alpha == True:
            self.density_estimator, self.density_estimator_2 = self.build_autoencoder()
        else:
            if config.use_cnn:
                self.density_estimator, self.cnn_model, self.autoencoder, self.input_autoencoder = self.build_autoencoder()
            else:
                self.density_estimator = self.build_autoencoder()
            if config.learn_alpha == 'heuristic':
                self.alpha = np.zeros([config.num_of_all_masks])
        self.train_end_epochs = []

    def generate_all_masks(self):
        all_masks = []
        all_pi = []

        for i_m in range(0,config.num_of_all_masks):

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
            
            mask = _detect_subsets(labels[i-1], labels[i])
            
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
                is_subset = _detect_subsets(labels[i-1], labels[i+1])
                
                
                zero_out_edge_nodes = np.where(np.sum(masks[i], axis=1) == 0)[0]
                for j in zero_out_edge_nodes:
                    t = np.random.randint(0,len(labels[i+1]))
                    k = np.random.choice( np.where(is_subset[:,t].reshape([-1]) > 0)[0] )

                    labels[i][j] = (labels[i-1][k]).copy()
                    rnd = np.random.uniform(0.,1.,labels[i+1][t].size)
                    labels[i][j] = np.union1d(labels[i][j], labels[i+1][t][rnd < mu[i-1]])
                    
                masks[i] = _detect_subsets(labels[i], labels[i+1])
                masks[i-1] = _detect_subsets(labels[i-1], labels[i])
        
        
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
        #generating subsets as 3d matrix 
        #subsets = np.random.randint(0, 2, (num_of_hlayer, hlayer_size, graph_size))
        labels = np.zeros([config.num_of_hlayer, config.hlayer_size], dtype=int)
        min_label = 0
        for ii in range(config.num_of_hlayer):
            labels[ii][:] = np.random.randint(min_label, config.graph_size, (config.hlayer_size))
            min_label = np.amin(labels[ii])
        #generating masks as 3d matrix
        #masks = np.zeros([num_of_hlayer,hlayer_size,hlayer_size])
        
        masks = []
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

        #first layer mask
        mask = np.zeros([config.graph_size, config.hlayer_size], dtype=np.float32)
        for j in range(0, config.hlayer_size):
            for k in range(0, config.graph_size):
                if (masking_method == 'orig'):
                    if (labels[0][j] >= pi[k]):
                        mask[k][j] = 1.0
                elif masking_method == 'min_related':
                    #if ((labels[0][j] >= pi[k]) and (labels[0][j] - related_size <= pi[k])):
                    if ((labels[0][j] >= pi[k]) and (min_related_pi[ labels[0][j] ] <= pi[k])):
                        mask[k][j] = 1.0
                else:
                    raise Exception("wrong masking method " + masking_method)
        masks.append(mask)
        
        #hidden layers mask   
        for i in range(1, config.num_of_hlayer):
            mask = np.zeros([config.hlayer_size, config.hlayer_size], dtype=np.float32)
            for j in range(0, config.hlayer_size):
                for k in range(0, config.hlayer_size):
                    if (masking_method == 'orig'):
                        if (labels[i][j] >= labels[i-1][k]):
                            mask[k][j] = 1.0
                    elif masking_method == 'min_related':
                        #if ((labels[i][j] >= labels[i-1][k]) and (labels[i][j] - related_size <= labels[i-1][k] )):
                        if ((labels[i][j] >= labels[i-1][k]) and (min_related_pi[labels[i][j]] <= labels[i-1][k] )):
                            mask[k][j] = 1.0
                    else:
                        raise Exception("wrong masking method " + masking_method)
    
            masks.append(mask)
        
        #last layer mask
        mask = np.zeros([config.hlayer_size, config.graph_size], dtype=np.float32)            
        for j in range(0, config.graph_size):
            for k in range(0, config.hlayer_size):
                if (masking_method == 'orig'):
                    if (pi[j] > labels[-1][k]):
                        mask[k][j] = 1.0
                elif (masking_method == 'min_related'):
                    #if ((pi[j] > labels[-1][k]) and (pi[j] - related_size <= labels[-1][k])):
                    if ((pi[j] > labels[-1][k]) and (min_related_pi[pi[j]] <= labels[-1][k])):
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
                        #ind = (((pi < pi[j]) & (pi >= (pi[j] - related_size))))
                        ind = (((pi < pi[j]) & (pi >= (min_related_pi[pi[j]]))))
                        if np.any(ind):
                            tmp_mask[ind, j] = 1.0
                    else:
                        raise Exception('Error' + str(config.direct_links))
            #print(tmp_mask.shape)
            mask = np.concatenate([mask, tmp_mask],axis=0)
            #mask = tmp_mask
            
        masks.append(mask)
        return masks


    def num_of_connections(self):
        cnt = 0
        for masks in self.all_masks:
            for mask in masks:
                cnt += np.sum(mask)
        return cnt

    def build_autoencoder(self):

        autoencoder_firstlayers = []

        for i in range(config.num_of_hlayer - 1):
            autoencoder_firstlayers.append(
                MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[i]), 'relu'))

        semiFinal_layer_mu = MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[-2]), 'relu')
        semiFinal_layer_sigma = MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[-2]), 'relu')
        semiFinal_layer_pi = MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[-2]), 'relu')

        final_layer_mu = []
        final_layer_logVar = []
        final_layer_logpi_unnormalized = []

        for i in range(config.num_mixture_components):
            if config.use_logit_preprocess:
                final_layer_mu.append(MaskedDenseLayer(config.graph_size, np.array(self.all_masks[-1]), 'linear'))
            else:
                final_layer_mu.append(MaskedDenseLayer(config.graph_size, np.array(self.all_masks[-1]), 'sigmoid'))
            final_layer_logVar.append(MaskedDenseLayer(config.graph_size, np.array(self.all_masks[-1]), 'linear'))
            final_layer_logpi_unnormalized.append(
                MaskedDenseLayer(config.graph_size, np.array(self.all_masks[-1]), 'linear'))

        def get_autoencode(inp, st):
            h = inp
            for layer in autoencoder_firstlayers:
                h = layer([h, st])
            h_mu = semiFinal_layer_mu([h, st])
            h_sigma = semiFinal_layer_sigma([h, st])
            h_pi = semiFinal_layer_pi([h, st])

            if config.direct_links:
                c_mu = Concatenate()([h_mu, inp])
                c_logVar = Concatenate()([h_sigma, inp])
                c_pi = Concatenate()([h_pi, inp])
            else:
                c_mu, c_logVar, c_pi = h_mu, h_sigma, h_pi

            f_mu, f_logVar, f_logpi_unnormalized = [], [], []
            for layer in final_layer_mu:
                f_mu.append(layer([c_mu, st]))
            for layer in final_layer_logVar:
                f_logVar.append(layer([c_logVar, st]))
            for layer in final_layer_logpi_unnormalized:
                f_logpi_unnormalized.append(layer([c_pi, st]))

            output = Concatenate()(f_mu + f_logVar + f_logpi_unnormalized)
            return output

        input_layer = Input(shape=(config.graph_size,))

        if config.use_cnn:

            def get_cnn(inp, last_activation):
                reshaped_input = Reshape([config.height, config.width // config.num_channels, config.num_channels])(inp)

                l1 = MaskedConvLayer(64, 3, 'relu')(reshaped_input)
                for block_num in range(30):
                    l2 = l1
                    for _ in range(2):
                        l2 = MaskedConvLayer(64, 3, 'relu')(l2)
                    l1 = Add()([l1, l2])
                last = MaskedConvLayer(config.num_channels, 5, last_activation)(l1)
                flattened = Flatten()(last)
                return flattened

            s_ = get_cnn(input_layer, last_activation='relu')
            t_ = get_cnn(input_layer, last_activation='sigmoid')
            nexp_layer = Lambda(lambda x: K.exp(-x))
            processed_input = Average()([Multiply()([nexp_layer(s_), input_layer]), t_])

            # cnn1 = MaskedConvLayer(20, 5, 'relu')(reshaped_input)
            # cnn2 = MaskedConvLayer(20, 5, 'relu')(cnn1)
            # cnn3 = MaskedConvLayer(20, 3, 'relu')(cnn2)
            # cnn4 = MaskedConvLayer(20, 5, 'relu')(cnn3)
            # cnn5 = MaskedConvLayer(config.num_channels, 5, 'sigmoid')(cnn4)
            # flattened = Flatten()(cnn5)
            # processed_input = Average()([flattened, input_layer])

        state = Input(shape=(1,), dtype="int32")

        if config.use_cnn:
            density_estimator = Model(inputs=[input_layer, state], outputs=[Concatenate()([get_autoencode(processed_input, state), processed_input, s_])])
            cnn_model = Model(inputs=[input_layer], outputs=[processed_input, s_, t_])
            input_autoencoder = Input(shape=(config.graph_size,))
            autoencoder = Model(inputs=[input_autoencoder, state], outputs=[get_autoencode(input_autoencoder, state)])
        else:
            density_estimator = Model(inputs=[input_layer, state], outputs=[get_autoencode(input_layer, state)])

        def normal_loss(y_true, y_pred):

            tmp_sz = config.num_mixture_components * config.graph_size
            mu_pred, logVar_pred, logpi_unnormalized_pred = y_pred[:, :tmp_sz], y_pred[:, tmp_sz:2*tmp_sz], y_pred[:, 2*tmp_sz:]

            # if config.use_uniform_noise_for_pmf:
            #     y_true = K.tile(y_true, [config.num_noisy_samples_per_sample, 1])
            #     y_true = y_true + K.random_uniform(K.shape(y_true)) / 256.0
            #     mu_pred = K.tile(mu_pred, [config.num_noisy_samples_per_sample, 1])
            #     logVar_pred = K.tile(logVar_pred, [config.num_noisy_samples_per_sample, 1])
            #     logpi_unnormalized_pred = K.tile(logpi_unnormalized_pred, [config.num_noisy_samples_per_sample, 1])

            mu_pred = K.reshape(mu_pred, [-1, config.num_mixture_components, config.graph_size])
            logVar_pred = K.reshape(logVar_pred, [-1, config.num_mixture_components, config.graph_size])
            logpi_unnormalized_pred = K.reshape(logpi_unnormalized_pred, [-1, config.num_mixture_components, config.graph_size])
            logpi_pred = logpi_unnormalized_pred - K.tile(K.logsumexp(logpi_unnormalized_pred, axis=1, keepdims=True), [1, config.num_mixture_components, 1])

            if config.min_var > 0:
                logVar_pred = K.log(K.exp(logVar_pred) + config.min_var)

            y_true_tiled = K.tile(K.expand_dims(y_true, 1), [1, config.num_mixture_components, 1])

            tmp = logpi_pred - 0.5 * logVar_pred - 0.5 * np.log(2*np.pi) - 0.5 * K.pow(y_true_tiled - mu_pred, 2) / K.exp(logVar_pred)

            tmp = K.logsumexp(tmp, axis=1)
            tmp = tmp - np.log(256)

            if config.use_logit_preprocess:
                k_ = K.exp(y_true) / (1 + K.exp(y_true))
                tmp += K.log(config.logit_scale/k_ + config.logit_scale/(1-k_))

            tmp = -K.sum(tmp, axis=1)

            # if config.use_uniform_noise_for_pmf:
            #     tmp = K.reshape(tmp, [config.num_noisy_samples_per_sample, -1])
            #     return K.logsumexp(tmp, axis=0) - np.log(config.num_noisy_samples_per_sample)
            return tmp


            # barrier = K.pow( 0.5 + keras.activations.sigmoid(10000 * (logVar_pred - 0.0025)), 10 )
            # return K.sum( 0.5 * K.pow(y_true - mu_pred, 2) / K.exp(logVar_pred) + logVar_pred/2 + barrier, axis=1)

        def normal_loss_use_cnn(y_true, y_pred):
            l = normal_loss(y_pred[:, -(2 * config.graph_size) : -(config.graph_size)], y_pred[:, : -(2 * config.graph_size)])
            s_ = y_pred[:, -(config.graph_size):]
            return l - K.sum( -s_ + np.log(0.5), axis=1)

        def _logistic_cdf(y, mu, s):
            y_altered = y + K.cast(y < 0, y.dtype) * -config.logistic_cdf_inf + K.cast(y > 255/256, y.dtype) * config.logistic_cdf_inf
            return 1 / (1 + K.exp(-1*(y_altered - mu)/s))

        def logistic_loss(y_true, y_pred):
            tmp_sz = config.num_mixture_components * config.graph_size
            mu_pred, logVar_pred, logpi_unnormalized_pred = y_pred[:, :tmp_sz], y_pred[:, tmp_sz:2 * tmp_sz], y_pred[:,
                                                                                                              2 * tmp_sz:]

            mu_pred = K.reshape(mu_pred, [-1, config.num_mixture_components, config.graph_size])
            logVar_pred = K.reshape(logVar_pred, [-1, config.num_mixture_components, config.graph_size])
            logpi_unnormalized_pred = K.reshape(logpi_unnormalized_pred,
                                                [-1, config.num_mixture_components, config.graph_size])
            logpi_pred = logpi_unnormalized_pred - K.tile(K.logsumexp(logpi_unnormalized_pred, axis=1, keepdims=True),
                                                          [1, config.num_mixture_components, 1])

            if config.min_var > 0:
                logVar_pred = K.log(K.exp(logVar_pred) + config.min_var)

            s_pred = K.exp( (logVar_pred + np.log(3) - 2 * np.log(np.pi)) / 2 )

            # s_pred = K.exp(logVar_pred/2) * np.sqrt(3) / np.p



            y_true_tiled = K.tile(K.expand_dims(y_true, 1), [1, config.num_mixture_components, 1])

            tmp = K.log(_logistic_cdf(y_true_tiled + 0.5/256, mu_pred, s_pred) - _logistic_cdf(y_true_tiled - 0.5/256, mu_pred, s_pred)) + logpi_pred

            tmp = -1 * K.logsumexp(tmp, axis=1)
            return K.sum(tmp, axis=1)

        def logistic_loss_use_cnn(y_true, y_pred):
            return logistic_loss(y_true, y_pred[:, : -(2 * config.graph_size)])


        if config.component_form == 'Gaussian':
            if config.use_cnn:
                density_estimator.compile(optimizer=config.optimizer, loss=normal_loss_use_cnn)
            else:
                density_estimator.compile(optimizer=config.optimizer, loss=normal_loss)
        elif config.component_form == 'logistic':
            if config.use_cnn:
                density_estimator.compile(optimizer=config.optimizer, loss=logistic_loss_use_cnn)
            else:
                density_estimator.compile(optimizer=config.optimizer, loss=logistic_loss)
        else:
            raise Exception('not implemented')

        if config.use_cnn:
            return density_estimator, cnn_model, autoencoder, input_autoencoder
        return density_estimator


    def _preprocess(self, data_clean):
        if config.use_uniform_noise_for_pmf:
            data = data_clean + np.random.rand(np.prod(data_clean.shape)).reshape(data_clean.shape) / 256
        else:
            data = data_clean.copy()
        if config.use_logit_preprocess:
            data = (((data * 2) - 1) * config.logit_scale + 1) / 2
            data = np.log(data) - np.log(1 - data)
        return data


    def fit(self, train_data_clean, validation_data_clean):
        cnt = 0
        best_loss = np.Inf
        best_weights = None
        for i in range(config.num_of_epochs):

            train_data = self._preprocess(train_data_clean)
            validation_data = self._preprocess(validation_data_clean)

            validation_size = validation_data.shape[0]
            reped_state_valid = (np.arange(validation_size * config.num_of_all_masks) / validation_size).astype(
                np.int32)
            reped_validdata = np.tile(validation_data, [config.num_of_all_masks, 1])

            if config.fast_train == True:
                train_size = train_data.shape[0]
                reped_state_train = np.random.randint(0, config.num_of_all_masks, train_size)
                reped_traindata = train_data
            else:
                train_size = train_data.shape[0]
                reped_state_train = (np.arange(train_size * config.num_of_all_masks) / train_size).astype(np.int32)
                reped_traindata = np.tile(train_data, [config.num_of_all_masks, 1])

            verbose = 1
            if verbose > 0:
                print('## Epoch:', i)
            train_history = self.density_estimator.fit(x=[reped_traindata, reped_state_train],
                                                       y=[reped_traindata],
                                                       epochs=1,
                                                       batch_size=config.batch_size,
                                                       shuffle=True,
                                                       validation_data=([reped_validdata, reped_state_valid],
                                                  [reped_validdata]),
                                                       verbose=verbose)
            val_loss = train_history.history['val_loss']
            print(type(val_loss), val_loss)
            if val_loss[-1] < best_loss:
                best_loss = val_loss[-1]
                cnt = 0
                best_weights = self.density_estimator.get_weights()
            else:
                cnt += 1
            if cnt >= config.patience:
                break

        if config.use_best_validated_weights:
            self.density_estimator.set_weights(best_weights)
        if i < config.num_of_epochs and verbose > 0:
            print('Epoch %05d: early stopping' % (config.num_of_epochs + 1))



    # def fit(self, train_data, validation_data):
    #
    #     early_stop = MyEarlyStopping(self.autoencoder, monitor='val_loss', min_delta=-0.0, patience=config.patience, verbose=1, mode='auto',
    #                                  train_end_epochs=self.train_end_epochs)
    #
    #     if config.fast_train == True:
    #         validation_size = validation_data.shape[0]
    #         reped_state_valid = (np.arange(validation_size * config.num_of_all_masks) / validation_size).astype(
    #             np.int32)
    #         reped_validdata = np.tile(validation_data, [config.num_of_all_masks, 1])
    #
    #         for i in range(0, config.fit_iter):
    #             train_size = train_data.shape[0]
    #             reped_state_train = np.random.randint(0, config.num_of_all_masks, train_size)
    #             reped_traindata = train_data
    #             self.autoencoder.fit(x=[reped_traindata, reped_state_train],
    #                                  y=[reped_traindata],
    #                                  epochs=config.num_of_epochs,
    #                                  batch_size=config.batch_size,
    #                                  shuffle=True,
    #                                  validation_data=([reped_validdata, reped_state_valid],
    #                                                   [reped_validdata]),
    #                                  callbacks=[early_stop],
    #                                  verbose=1)
    #
    #     else:
    #         train_size = train_data.shape[0]
    #         reped_state_train = (np.arange(train_size * config.num_of_all_masks) / train_size).astype(np.int32)
    #         reped_traindata = np.tile(train_data, [config.num_of_all_masks, 1])
    #         validation_size = validation_data.shape[0]
    #         reped_state_valid = (np.arange(validation_size*config.num_of_all_masks)/validation_size).astype(np.int32)
    #         reped_validdata = np.tile(validation_data, [config.num_of_all_masks, 1])
    #
    #         for i in range(0, config.fit_iter):
    #             self.autoencoder.fit(x=[reped_traindata, reped_state_train],
    #                                  y=[reped_traindata],
    #                                  epochs=config.num_of_epochs,
    #                                  batch_size=config.batch_size,
    #                                  shuffle=True,
    #                                  validation_data=([reped_validdata, reped_state_valid],
    #                                                   [reped_validdata]),
    #                                  callbacks=[early_stop],
    #                                  verbose=1)

    def predict(self, test_data, pixelwise=False):
        first = 0
        res = []
        while first < test_data.shape[0]:
            last = min(test_data.shape[0], first+config.batch_size)
            res.append(self._predict(test_data[first:last], pixelwise))
            first = last
        return np.concatenate(res, axis=0)

    def _predict(self, test_data, pixelwise=False):

        test_data_preprocessed = self._preprocess(test_data)

        if config.use_cnn:
            test_data, s_, _ = self.cnn_model.predict(test_data_preprocessed)
            model = self.autoencoder
        else:
            test_data = test_data_preprocessed
            model = self.density_estimator

        def _logistic_cdf_numpy(y, mu, s):
            y_altered = y.copy()
            y_altered[y < 0] = -np.inf
            #### 0-255
            y_altered[y > 255/256] = np.inf
            # y_altered[y > 255] = np.inf
            return 1 / (1 + np.exp(-1 * (y_altered - mu) / s))

        print('predict start')
        test_size = test_data.shape[0]
        if not pixelwise:
            all_masks_log_probs = np.zeros([config.num_of_all_masks, test_size])
        else:
            all_masks_log_probs = np.zeros([config.num_of_all_masks, test_size, config.graph_size])

        # if config.use_uniform_noise_for_pmf:
        #     noisy_test_data = np.tile(test_data, [config.num_noisy_samples_per_sample, 1])
        #     noisy_test_data = noisy_test_data + np.random.rand(np.prod(noisy_test_data.shape)).reshape(noisy_test_data.shape) / 256

        for j in range(config.num_of_all_masks):
            made_predict = model.predict([test_data, j * np.ones([test_size, 1])])#.reshape(1, hlayer_size, graph_size)]

            tmp_sz = config.num_mixture_components * config.graph_size
            made_predict_mu = made_predict[ :, :tmp_sz]
            made_predict_logVar = made_predict[ :, tmp_sz:2*tmp_sz]
            made_predict_logpi_unnormalized = made_predict[ :, 2*tmp_sz:3*tmp_sz]

            # if config.use_uniform_noise_for_pmf:
            #     made_predict_mu = np.tile(made_predict_mu, [config.num_noisy_samples_per_sample, 1])
            #     made_predict_logVar = np.tile(made_predict_logVar, [config.num_noisy_samples_per_sample, 1])
            #     made_predict_logpi_unnormalized = np.tile(made_predict_logpi_unnormalized, [config.num_noisy_samples_per_sample, 1])

            made_predict_mu = np.reshape(made_predict_mu, [-1, config.num_mixture_components, config.graph_size])
            made_predict_logVar = np.reshape(made_predict_logVar, [-1, config.num_mixture_components, config.graph_size])
            made_predict_logpi_unnormalized = np.reshape(made_predict_logpi_unnormalized, [-1, config.num_mixture_components, config.graph_size])
            made_predict_logpi = made_predict_logpi_unnormalized - np.tile(logsumexp(made_predict_logpi_unnormalized, axis=1, keepdims=True), [1, config.num_mixture_components, 1])

            if config.min_var > 0:
                made_predict_logVar = np.log(np.exp(made_predict_logVar) + config.min_var)

            test_data_tiled = np.tile(np.expand_dims(test_data, 1), [1, config.num_mixture_components, 1])
            # if config.use_uniform_noise_for_pmf:
            #     test_data_tiled = np.tile(np.expand_dims(noisy_test_data, 1), [1, config.num_mixture_components, 1])
            # else:
            #     test_data_tiled = np.tile(np.expand_dims(test_data, 1), [1, config.num_mixture_components, 1])

            if config.component_form == 'Gaussian':
                tmp = -0.5 * (test_data_tiled - made_predict_mu)**2 / np.exp(made_predict_logVar) - made_predict_logVar/2 - np.log(2*np.pi)/2 + made_predict_logpi
            elif config.component_form == 'logistic':
                made_predict_s = np.exp((made_predict_logVar + np.log(3) - 2 * np.log(np.pi)) / 2)
                #made_predict_s = np.exp(made_predict_logVar / 2) * np.sqrt(3) / np.pi
                #### 0-255
                tmp = np.log(
                    _logistic_cdf_numpy(test_data_tiled + 0.5 / 255, made_predict_mu, made_predict_s) -
                    _logistic_cdf_numpy(test_data_tiled - 0.5 / 255, made_predict_mu, made_predict_s)) + made_predict_logpi
                # tmp = np.log(
                #     _logistic_cdf_numpy(test_data_tiled*255 + 0.5, made_predict_mu*255, made_predict_s) -
                #     _logistic_cdf_numpy(test_data_tiled*255 - 0.5, made_predict_mu*255, made_predict_s)) + made_predict_logpi
            else:
                raise Exception('not implemented')
            log_probs = logsumexp(tmp, axis=1)
            if config.use_cnn:
                if config.component_form == 'Gaussian':
                    log_probs = log_probs + ( -s_ + np.log(0.5))
            if config.component_form == 'Gaussian':
                log_probs = log_probs - np.log(256)

            # eps = 0.00001
            # log_probs[log_probs < np.log(eps)] = np.log(eps)
            # log_probs[log_probs > np.log(1 - eps)] = np.log(1 - eps)

            # if config.use_uniform_noise_for_pmf:
            #     log_probs = log_probs.reshape([config.num_noisy_samples_per_sample, test_size, config.graph_size])
            #     log_probs = logsumexp(log_probs, axis=0) - np.log(config.num_noisy_samples_per_sample)

            if config.use_logit_preprocess:
                k_ = np.exp(test_data_preprocessed) / (1 + np.exp(test_data_preprocessed))
                log_probs += np.log(config.logit_scale / k_ + config.logit_scale / (1 - k_))

            if not pixelwise:
                made_log_prob = np.sum(log_probs, axis=1)
                all_masks_log_probs[j][:] = made_log_prob
            else:
                all_masks_log_probs[j,:,:] = log_probs


        #res = np.log(np.mean(probs, axis=0))
        res = logsumexp(all_masks_log_probs, axis=0) - np.log(config.num_of_all_masks)
        print('predict finish')
        return res

    ## implemented just for 1 Gaussian (not mixture of Gaussians)
    def generate(self, n):
        if config.use_cnn:
            model = self.autoencoder
        else:
            model = self.density_estimator
        mask_index = np.random.randint(0,config.num_of_all_masks,n)
        generated_samples = np.zeros([n,config.graph_size])
        all_pi_nparray = np.concatenate([pi.reshape([1,-1]) for pi in self.all_pi], axis=0)
        for i in range(config.graph_size):
            ind = (all_pi_nparray[mask_index,:] == i)
            pred = model.predict([generated_samples, mask_index.reshape([-1, 1])])
            mu = pred[ :, :config.graph_size][ind]
            if config.min_var > 0:
                logVar = np.log(np.exp(pred[ :, config.graph_size:2*config.graph_size][ind]) + config.min_var)
            else:
                logVar = pred[:, config.graph_size:2 * config.graph_size][ind]
            generated_samples[ind] = np.random.normal(mu, np.exp(logVar/2))
        if not config.use_cnn:
            return generated_samples
        generated_pixels = np.zeros(generated_samples.shape)
        for i in range(config.graph_size):
            _, s_, t_ = self.cnn_model.predict(generated_pixels)
            generated_pixels[:, i] = (2 * generated_samples[:, i] - t_[:,i]) / np.exp(-s_[:,i])
        return generated_pixels


    def reaching_dimesions_num(self):
        results = np.zeros([config.num_of_all_masks, config.graph_size])
        for mask_i in range(config.num_of_all_masks):
            for i in range(config.graph_size):
                #print(mask_i, i)
                mark = np.zeros([config.num_of_hlayer+2, config.hlayer_size])
                mark[config.num_of_hlayer+1, i] = 1
                results[mask_i, i] = len(self._get_reaching_dimensions(i, config.num_of_hlayer+1, mark, mask_i))
        return results

    def _get_reaching_dimensions(self, i, l, mark, mask_i):
        mark[l, i] = 1
        if l == 0:
            return [i]
        result = []
        adj = self.all_masks[l-1][mask_i]
        k = adj.shape[0]
        if k > config.hlayer_size:
            k = config.hlayer_size
            adj = adj[:k, :]
        for j in range(k):
            if mark[l-1, j] == 0 and adj[j, i] == 1:
                result = result + self._get_reaching_dimensions(j, l-1, mark, mask_i)
        return result
