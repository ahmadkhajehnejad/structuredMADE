import numpy as np
from keras.models import Model
from keras.layers import Input, Concatenate, Reshape
import config
from keras import backend as K
from made_utils import MaskedDenseLayer, MyEarlyStopping, MaskedDenseLayerMultiMasks,\
    LikelihoodConvexCombinationLayer, negative_log_likelihood_loss #, log_sum_exp
from dataset import get_data_structure
#from keras import optimizers
import keras
import grid_orders
import bfs_orders
#import threading
if config.use_multiprocessing:
    from multiprocessing import Process, Queue
#import sys
from scipy.misc import logsumexp
from sklearn.linear_model import LogisticRegression


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
    def __init__(self):
        self.masking_method = config.algorithm
        if (self.masking_method in ['Q_restricted', 'random_Q_restricted' , 'ensemble_Q_restricted_and_orig', 'min_related']) or \
                (config.random_dimensions_order in ['bfs']) or \
                (str(config.random_dimensions_order).endswith('bfs-random')):
            parameters = get_data_structure()
            self.adjacency_matrix = parameters['adjacency_matrix']
        self.all_masks, self.all_pi = self.generate_all_masks()

        #print('avg reaching sizes: ', np.mean(self.reaching_dimesions_num()))

        if config.learn_alpha == True:
            self.autoencoder, self.autoencoder_2 = self.build_autoencoder()
        else:
            self.autoencoder = self.build_autoencoder()
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

        input_layer = Input(shape=(config.graph_size,))
        state = Input(shape=(1,), dtype="int32")

        hlayer = MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[0]), 'relu')( [input_layer, state] )
        for i in range(1,config.num_of_hlayer):
            hlayer = MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[i]), 'relu')( [hlayer, state] )
        if config.direct_links:
            clayer = Concatenate()([hlayer, input_layer])
            #clayer = input_layer
            output_layer = MaskedDenseLayer(config.graph_size, np.array(self.all_masks[-1]), 'sigmoid')( [clayer, state] )
        else:
            output_layer = MaskedDenseLayer(config.graph_size, np.array(self.all_masks[-1]), 'sigmoid')( [hlayer, state] )
        autoencoder = Model(inputs=[input_layer, state], outputs=[output_layer])


        '''
        def reg_(W):
            return K.sum(K.pow(W, 2)) # K.mean(K.sqrt(K.sum(K.pow(W, 2), axis=0)))

        autoencoder.reg_function = reg_
        autoencoder.reg_coef = 1./(config.batch_size * config.graph_size)

        def regularized_binary_cross_entropy_loss(y_true, y_pred):
            W = autoencoder.layers[-1].trainable_weights[0]
            return keras.losses.binary_crossentropy(y_true, y_pred) + autoencoder.reg_coef * autoencoder.reg_function(W)

        autoencoder.compile(optimizer=config.optimizer, loss=regularized_binary_cross_entropy_loss)
        '''
        autoencoder.compile(optimizer=config.optimizer, loss='binary_crossentropy')

        if config.learn_alpha == True:

            input_layer_2 = Input(shape=(config.num_of_all_masks * config.graph_size,))


            reshape_layer_2 = Reshape(target_shape=(config.num_of_all_masks, config.graph_size))(input_layer_2)

            hlayer_2 = MaskedDenseLayerMultiMasks(config.hlayer_size, np.array(self.all_masks[0]), 'relu')([reshape_layer_2])
            for i in range(1, config.num_of_hlayer):
                hlayer_2 = MaskedDenseLayerMultiMasks(config.hlayer_size, np.array(self.all_masks[i]), 'relu')([hlayer_2])
            if config.direct_links:
                clayer_2 = Concatenate()([hlayer_2, reshape_layer_2])
                pre_output_layer_2 = MaskedDenseLayerMultiMasks(config.graph_size, np.array(self.all_masks[-1]), 'sigmoid')([clayer_2])
            else:
                pre_output_layer_2 = MaskedDenseLayerMultiMasks(config.graph_size, np.array(self.all_masks[-1]), 'sigmoid')([hlayer_2])

            output_layer_2 = LikelihoodConvexCombinationLayer(config.num_of_all_masks)([pre_output_layer_2, reshape_layer_2])
            autoencoder_2 = Model(inputs=[input_layer_2], outputs=[output_layer_2])

            autoencoder_2.compile(optimizer=config.optimizer, loss=negative_log_likelihood_loss)

            return autoencoder, autoencoder_2

        else:
            return autoencoder


    def logReg_pretrain(self, train_data):
        print('training start')
        print('logistic regressions start')
        d = train_data.shape[1]

        pi = self.all_pi[0]
        q = np.zeros([d], dtype=int)
        for i in range(d):
            q[pi[i]] = i
        self.unique_label = [None] * d
        if np.unique(train_data[:, q[0]]).size == 1:
            self.unique_label[0] = train_data[0, q[0]]
        else:
            self.mu_0 = np.sum(train_data[:, q[0]] == 1) / train_data.shape[0]
        clf = [None] * d
        for i in range(1, d):
            if np.unique(train_data[:, q[i]]).size == 1:
                self.unique_label[i] = train_data[0, q[i]]
            else:
                clf[i] = LogisticRegression(random_state=0, solver='liblinear', penalty='l1',  C=1,
                                            multi_class='ovr').fit(train_data[:, q[:i]].reshape([-1, i]),
                                                                   train_data[:, q[i]])

        print('logistic regressions finish')

        W = np.zeros([d, d])
        b_0 = np.zeros([d])
        for i in range(d):
            if self.unique_label[i] == 0:
                b_0[q[i]] = -20
            elif self.unique_label[i] == 1:
                b_0[q[i]] = 20
            else:
                if i == 0:
                    b_0[q[i]] = np.log(self.mu_0)
                else:
                    if clf[i].classes_[0] == 1:
                        W[q[:i], q[i]] = -clf[i].coef_
                    else:
                        W[q[:i], q[i]] = clf[i].coef_
                    b_0[q[i]] = clf[i].intercept_
        W_all = np.zeros(self.autoencoder.layers[-1].get_weights()[0].shape)
        W_all[ -d:, :] = W
        self.autoencoder.layers[-1].set_weights([W_all, b_0.reshape([1, -1])])
        print('pre-training finish')



    def fit(self, train_data, validation_data):

        if config.logReg_pretrain == True:
            self.logReg_pretrain(train_data)

        early_stop = MyEarlyStopping(self.autoencoder, monitor='val_loss', min_delta=-0.0, patience=config.patience, verbose=1, mode='auto',
                                     train_end_epochs=self.train_end_epochs)

        if config.fast_train == True:
            validation_size = validation_data.shape[0]
            reped_state_valid = (np.arange(validation_size * config.num_of_all_masks) / validation_size).astype(
                np.int32)
            reped_validdata = np.tile(validation_data, [config.num_of_all_masks, 1])

            for i in range(0, config.fit_iter):
                train_size = train_data.shape[0]
                reped_state_train = np.random.randint(0, config.num_of_all_masks, train_size)
                reped_traindata = train_data
                self.autoencoder.fit(x=[reped_traindata, reped_state_train],
                                     y=[reped_traindata],
                                     epochs=config.num_of_epochs,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     validation_data=([reped_validdata, reped_state_valid],
                                                      [reped_validdata]),
                                     callbacks=[early_stop],
                                     verbose=1)

        else:
            train_size = train_data.shape[0]
            reped_state_train = (np.arange(train_size * config.num_of_all_masks) / train_size).astype(np.int32)
            reped_traindata = np.tile(train_data, [config.num_of_all_masks, 1])
            validation_size = validation_data.shape[0]
            reped_state_valid = (np.arange(validation_size*config.num_of_all_masks)/validation_size).astype(np.int32)
            reped_validdata = np.tile(validation_data, [config.num_of_all_masks, 1])

            for i in range(0, config.fit_iter):
                self.autoencoder.fit(x=[reped_traindata, reped_state_train],
                                     y=[reped_traindata],
                                     epochs=config.num_of_epochs,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     validation_data=([reped_validdata, reped_state_valid],
                                                      [reped_validdata]),
                                     callbacks=[early_stop],
                                     verbose=1)
              
        if config.learn_alpha == True:

            for l in range(config.num_of_hlayer):
                self.autoencoder_2.get_layer(index=l+2).set_weights(self.autoencoder.get_layer(index=l+2).get_weights())

            self.autoencoder_2.get_layer(index=config.num_of_hlayer + 3).set_weights(
                self.autoencoder.get_layer(index=config.num_of_hlayer + 3).get_weights())


            for i in range(0, config.fit_iter):
                self.autoencoder_2.fit(x=np.tile( train_data.reshape([-1, 1, config.graph_size]), [1, config.num_of_all_masks, 1]).reshape([-1, config.num_of_all_masks*config.graph_size]),
                                       y=np.zeros((train_data.shape[0], 1)),
                                       epochs=config.num_of_epochs,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       validation_data=(np.tile(validation_data.reshape([-1, 1, config.graph_size]), [1, config.num_of_all_masks, 1]).reshape([-1, config.num_of_all_masks*config.graph_size]), \
                                                        np.zeros((validation_data.shape[0],1))),
                                       callbacks=[early_stop],
                                       verbose=0)


            #alpha = np.exp(K.eval(self.autoencoder_2._layers[-1]._trainable_weights[0]))
            alpha = np.exp(self.autoencoder_2.get_layer(index=-1).get_weights()[0])
            print('alpha: ', alpha/np.sum(alpha))
        elif config.learn_alpha == 'heuristic':
            _x = np.concatenate([train_data, validation_data], axis=0)
            _n = train_size + validation_size
            predicted_probs = np.zeros([_n, config.num_of_all_masks])
            for s in range(config.num_of_all_masks):
                _y = self.autoencoder.predict(x=[_x, s*np.ones([_n,1])])
                predicted_probs[:, s] = np.prod(np.multiply(np.power(_y, _y), np.power(1- _y, 1- _x)), axis=1)
            amx = np.argmax(predicted_probs, axis=1)
            for s in range(config.num_of_all_masks):
                self.alpha[s] = np.sum(amx == s)
            self.alpha = (self.alpha + 1) / np.sum(self.alpha + 1)
            print('alpha: ', self.alpha)


    def predict(self, test_data):
        print('predict start')
        if config.learn_alpha == True:
            probs = self.autoencoder_2.predict(np.tile(test_data.reshape([-1,1,config.graph_size]), [1,config.num_of_all_masks,1]).reshape([-1, config.num_of_all_masks*config.graph_size]))
            res = np.log(probs)
        else:
            test_size = test_data.shape[0]
            #probs = np.zeros([config.num_of_all_masks, test_size])
            log_probs = np.zeros([config.num_of_all_masks, test_size])
            for j in range(config.num_of_all_masks):
                made_predict = self.autoencoder.predict([test_data, j * np.ones([test_size,1])])#.reshape(1, hlayer_size, graph_size)]
                eps = 0.00001
                made_predict[made_predict < eps] = eps
                made_predict[made_predict > 1- eps] = 1-eps

                #corrected_probs = np.multiply(np.power(made_predict, test_data),
                                # np.power(np.ones(made_predict.shape) - made_predict, np.ones(test_data.shape) - test_data))
                # made_prob = np.prod(corrected_probs, 1)
                #probs[j][:] = made_prob

                corrected_log_probs = (np.log(made_predict) * test_data) + (np.log(1 - made_predict) * (1 - test_data))
                made_log_prob = np.sum(corrected_log_probs, axis=1)
                log_probs[j][:] = made_log_prob

            if config.learn_alpha == 'heuristic':
                #res = np.log(np.matmul(self.alpha.reshape([1,-1]), probs)).reshape([-1])
                res = np.log(np.matmul(self.alpha.reshape([1, -1]), np.exp(log_probs))).reshape([-1])
            else:
                #res = np.log(np.mean(probs, axis=0))
                res = logsumexp(log_probs, axis=0) - np.log(config.num_of_all_masks)
        print('predict finish')
        return res

    def generate(self, n):
        if config.learn_alpha != False:
            raise Exception('Not implemented')
        mask_index = np.random.randint(0,config.num_of_all_masks,n)
        generated_samples = np.zeros([n,config.graph_size])
        all_pi_nparray = np.concatenate([pi.reshape([1,-1]) for pi in self.all_pi], axis=0)
        for i in range(config.graph_size):
            ind = (all_pi_nparray[mask_index,:] == i)
            mu = self.autoencoder.predict([generated_samples, mask_index.reshape([-1,1])])[ind]
            generated_samples[ind] = np.array(np.random.rand(n) < mu, dtype=np.float32)
        return generated_samples



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
