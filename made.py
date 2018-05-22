import numpy as np
from keras.models import Model
from keras.layers import Input
import config
from made_utils import MaskedDenseLayer, MyEarlyStopping, log_sum_exp
from dataset import get_data_structure


def _spread(current_node, root_node, visited, adj):
        visited[current_node]=True
        if current_node < root_node:
            return
        for nei in range(adj.shape[0]):
            if adj[current_node,nei] == 1 and not visited[nei]:
                _spread(nei, root_node, visited, adj)
                
                
    
def _make_Q(adj):
    n = adj.shape[0]
    Q = [None] * n
    for i in range(n):
        visited = np.zeros([n],dtype=bool)
        _spread(current_node = i, root_node = i, visited = visited, adj=adj)
        visited[i:] = False
        Q[i] = np.where(visited)[0]
    return Q    


class MADE:
    def __init__(self):
        self.masking_method = config.algorithm
        if self.masking_method == 'Q_restricted':
            parameters = get_data_structure()
            self.adjacency_matrix = parameters['adjacency_matrix']
        self.all_masks = self.generate_all_masks()
        
        self.autoencoder = self.build_autoencoder()
        self.train_end_epochs = []

    def generate_all_masks(self):
        if self.masking_method == 'Q_restricted':
            all_masks = []
            ### We can change it
            mu = [(i+1)/(config.num_of_hlayer+1) for i in range(config.num_of_hlayer)]
            
            for i_m in range(0,config.num_of_all_masks):
        
                
                ###  We can add different orderings here
                Q = _make_Q(self.adjacency_matrix)
                
                
                
                labels = []
                for i in range(config.num_of_hlayer):
                    labels.append([None] * config.hlayer_size)
                labels.insert(0,[np.array([i]) for i in range(config.graph_size)])
                labels.append( [Q[i] for i in range(config.graph_size)])

                for i in range(1,config.num_of_hlayer+1):
                    corresp_Q_inds = np.random.randint(0,config.graph_size,config.hlayer_size)
                    rnd_prm = np.random.permutation(config.hlayer_size)
                    corresp_Q_inds[ rnd_prm[:config.graph_size] ] = np.arange(0,config.graph_size)
                    
                    for j in range(config.hlayer_size):
                        
                        corresp_Q = Q[corresp_Q_inds[j]]
                        while corresp_Q.size == 0:
                            corresp_Q = Q[np.random.randint(0,config.graph_size)]

                        tmp = [k for k in range(len(labels[i-1])) if np.intersect1d(labels[i-1][k], corresp_Q).size == np.unique(labels[i-1][k]).size]
                        pr = np.random.choice(tmp)
                        
                        
                        labels[i][j] = (labels[i-1][pr]).copy()
                                                
                        rnd = np.random.uniform(0.,1.,corresp_Q.size)
                                                
                        labels[i][j] = np.union1d(labels[i][j], corresp_Q[rnd < mu[i-1]])
                        
                
                #cnt_masks = 0
                masks = []
                for i in range(1,len(labels)):
                    mask = np.zeros([len(labels[i-1]), len(labels[i])], dtype=np.float32)
                    for k in range(len(labels[i-1])):
                        for j in range(len(labels[i])):
                            mask[k][j] = (np.intersect1d(labels[i-1][k], labels[i][j]).size == np.unique(labels[i-1][k]).size)
                            #if mask[k][j]:
                            #    cnt_masks += 1
                    masks.append(mask)
        
                #print('-- ' + str(cnt_masks))
                all_masks.append(masks)
        else:
            all_masks = []
            for i_m in range(0,config.num_of_all_masks):
                #generating subsets as 3d matrix 
                #subsets = np.random.randint(0, 2, (num_of_hlayer, hlayer_size, graph_size))
                labels = np.zeros([config.num_of_hlayer, config.hlayer_size], dtype=np.float32)
                min_label = 0
                for ii in range(config.num_of_hlayer):
                    labels[ii][:] = np.random.randint(min_label, config.graph_size, (config.hlayer_size))
                    min_label = np.amin(labels[ii])
                #generating masks as 3d matrix
                #masks = np.zeros([num_of_hlayer,hlayer_size,hlayer_size])
                
                masks = []
        #        if (algo == 'orig'):
        #            pi = np.random.permutation(graph_size)
        #            #pi = np.array(range(graph_size))
        #        else:
        #            pi = np.array(range(graph_size))
                #first layer mask
                mask = np.zeros([config.graph_size, config.hlayer_size], dtype=np.float32)
                for j in range(0, config.hlayer_size):
                    for k in range(0, config.graph_size):
                        if (self.masking_method == 'orig'):
                            if (labels[0][j] >= k): #and (pi[k] >= labels[0][j]-width)):
                                mask[k][j] = 1.0
                        elif self.masking_method == 'min_related':
                            if ((labels[0][j] >= k) and (k - config.related_size <= labels[0][j])): #cant use permutation in our approach
                                mask[k][j] = 1.0
                        else:
                            print("wrong masking method " + self.masking_method)
                masks.append(mask)
                
                #hidden layers mask   
                for i in range(1, config.num_of_hlayer):
                    mask = np.zeros([config.hlayer_size, config.hlayer_size], dtype=np.float32)
                    for j in range(0, config.hlayer_size):
                        for k in range(0, config.hlayer_size):
                            if (self.masking_method == 'orig'):
                                if (labels[i][j] >= labels[i-1][k]): #and (labels[i][j] >= labels[i-1][k]-width)):
                                    mask[k][j] = 1.0
                            elif self.masking_method == 'min_related':
                                if ((labels[i][j] >= labels[i-1][k]) and (labels[i][j] - config.related_size <= labels[i-1][k] )):
                                    mask[k][j] = 1.0
                            else:
                                print("wrong masking method " + self.masking_methdo)

                    masks.append(mask)
                
                #last layer mask
                mask = np.zeros([config.hlayer_size, config.graph_size], dtype=np.float32)
                #last_layer_label = np.random.randint(0, 4, graph_size)
                for j in range(0, config.graph_size):
                    for k in range(0, config.hlayer_size):
                        if (self.masking_method == 'orig'):
                            if (j > labels[-1][k]): #and (j >= labels[-1][k]-width)):
                                mask[k][j] = 1.0
                        elif (self.masking_method == 'min_related'):
                            if ((j > labels[-1][k]) and (j - config.related_size <= labels[-1][k])):
                                mask[k][j] = 1.0
                        else:
                            print("wrong masking method " + self.masking_method)
                masks.append(mask)
                all_masks.append(masks)
        
        
        swapped_all_masks = []
        for i in range(config.num_of_hlayer+1):
            swapped_masks = []
            for j in range(config.num_of_all_masks):
                swapped_masks.append(all_masks[j][i])
            swapped_all_masks.append(swapped_masks)
            
        #all_masks = [[x*1.0 for x in y] for y in all_masks]
        return swapped_all_masks

    def num_of_connections(self):
        cnt = 0
        for masks in self.all_masks:
            for mask in masks:
                cnt += np.sum(mask)
        return cnt

    def build_autoencoder(self):

        input_layer = Input(shape=(config.graph_size,))
        state = Input(shape=(1,), dtype = "int32")
    
        hlayer = MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[0]), 'relu')( [input_layer, state] )
        for i in range(1,config.num_of_hlayer):
            hlayer = MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[i]), 'relu')( [hlayer, state] )
        output_layer = MaskedDenseLayer(config.graph_size, np.array(self.all_masks[-1]), 'sigmoid')( [hlayer, state] )
        autoencoder = Model(inputs=[input_layer, state], outputs=[output_layer])
        
        autoencoder.compile(optimizer=config.optimizer, loss='binary_crossentropy')
        return autoencoder
    
    def fit(self, train_data, validation_data):
        early_stop = MyEarlyStopping(monitor='val_loss', min_delta=0, patience=config.patience, verbose=1, mode='auto', train_end_epochs = self.train_end_epochs)
        
        train_size = train_data.shape[0]
        validation_size = validation_data.shape[0]
        reped_state_train = (np.arange(train_size*config.num_of_all_masks)/train_size).astype(np.int32)
        reped_state_valid = (np.arange(validation_size*config.num_of_all_masks)/validation_size).astype(np.int32)
        reped_traindata = np.tile(train_data, [config.num_of_all_masks, 1])
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
    def predict(self, test_data):
        test_size = test_data.shape[0]
        probs = np.zeros([config.num_of_all_masks, test_size])
        #log_probs = np.zeros([config.num_of_all_masks, test_size])
        for j in range(config.num_of_all_masks):
            made_predict = self.autoencoder.predict([test_data, j * np.ones([test_size,1])])#.reshape(1, hlayer_size, graph_size)]
            
            corrected_probs = np.multiply(np.power(made_predict, test_data), 
                            np.power(np.ones(made_predict.shape) - made_predict, np.ones(test_data.shape) - test_data))
            
            #made_log_prob = np.sum(np.log(corrected_probs), axis=1)
            #log_probs[j][:] = made_log_prob
            
            made_prob = np.prod(corrected_probs, 1)
            probs[j][:] = made_prob
            
        #res = log_sum_exp(log_probs, axis=0) - np.log(config.num_of_all_masks)            
        res = np.log(np.mean(probs, axis=0))
        return res
