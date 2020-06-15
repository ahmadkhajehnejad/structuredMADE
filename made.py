import numpy as np
from keras.models import Model
from keras.layers import Input, Concatenate, Reshape, Flatten, Add, Average, Multiply, Lambda
import config
import keras
from keras import backend as K

from utils.made_base import MADE_base
from utils.made_utils import MaskedDenseLayer, MyEarlyStopping, MaskedConvLayer
from dataset import get_data_structure
#from keras import optimizers
import keras
from utils import grid_orders
import bfs_orders
#import threading
from utils.masking_utils import _make_Q, _detect_subsets

if config.use_multiprocessing:
    from multiprocessing import Process, Queue
#import sys
from sklearn.linear_model import LogisticRegression
from functools import reduce
from scipy.special import logsumexp
import tensorflow as tf
from utils.logistic_functions import logistic_loss


class MADE(MADE_base):
    def __init__(self, all_masks=None, all_pi=None):
        super(MADE, self).__init__(all_masks, all_pi)
        self.density_estimator = self.build_autoencoder()

    def build_autoencoder(self):

        activation = 'relu' # 'tanh' # 'elu'

        autoencoder_firstlayers = []

        for i in range(config.num_of_hlayer - 1):
            autoencoder_firstlayers.append(
                MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[i]), activation))

        semiFinal_layer_mu = MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[-2]), activation)
        semiFinal_layer_sigma = MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[-2]), activation)
        semiFinal_layer_pi = MaskedDenseLayer(config.hlayer_size, np.array(self.all_masks[-2]), activation)

        final_layer_mu = []
        final_layer_logVar = []
        final_layer_logpi_unnormalized = []

        for i in range(config.num_mixture_components):
            if config.use_logit_preprocess:
                final_layer_mu.append(MaskedDenseLayer(config.graph_size, np.array(self.all_masks[-1]), 'linear'))
            else:
                final_layer_mu.append(MaskedDenseLayer(config.graph_size, np.array(self.all_masks[-1]), 'linear'))
            final_layer_logVar.append(MaskedDenseLayer(config.graph_size, np.array(self.all_masks[-1]), 'linear'))
            final_layer_logpi_unnormalized.append(
                MaskedDenseLayer(config.graph_size, np.array(self.all_masks[-1]), 'linear'))

        def get_autoencode(inp1, inp2, st):
            h = inp1
            for layer in autoencoder_firstlayers:
                h = layer([h, st])
            h_mu = semiFinal_layer_mu([h, st])
            h_sigma = semiFinal_layer_sigma([h, st])
            h_pi = semiFinal_layer_pi([h, st])

            if config.direct_links:
                c_mu = Concatenate()([h_mu, inp2])
                c_logVar = Concatenate()([h_sigma, inp2])
                c_pi = Concatenate()([h_pi, inp2])
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

            def get_cnn(inp): #, last_activation):
                reshaped_input = Reshape([config.height, config.width // config.num_channels, config.num_channels])(inp)

                l1 = MaskedConvLayer(64, 3, 'relu')(reshaped_input)
                for block_num in range(30):
                    l2 = l1
                    for _ in range(2):
                        l2 = MaskedConvLayer(64, 3, 'relu')(l2)
                    l1 = Add()([l1, l2])
                # last = MaskedConvLayer(config.num_channels, 5, last_activation)(l1)
                last = l1
                flattened = Flatten()(last)
                return flattened

            processed_input = get_cnn(input_layer)
        else:
            processed_input = input_layer

        state = Input(shape=(1,), dtype="int32")
        output = get_autoencode(processed_input, input_layer, state)

        density_estimator = Model(inputs=[input_layer, state], outputs=[output])

        if config.component_form == 'logistic':
            density_estimator.compile(optimizer=config.optimizer, loss=logistic_loss)
        else:
            raise Exception('not implemented')

        return density_estimator


    def _preprocess(self, raw_data):
        if config.scale_negOne_to_posOne:
            data = ((raw_data * 256) / 255) * 2 - 1
        else:
            data = (raw_data * 256) / 255
        return data


    def _get_permuted_data(self, data, state):
        n = data.shape[0]
        data_permuted = np.zeros(data.shape)
        for i in range(n):
            data_permuted[i, :] = data[i, np.argsort(self.all_pi[state[i]])]
        return data_permuted

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

            permuted_reped_traindata = self._get_permuted_data(reped_traindata, reped_state_train)
            permuted_reped_validdata = self._get_permuted_data(reped_validdata, reped_state_valid)

            verbose = 1
            if verbose > 0:
                print('## Epoch:', i)
            train_history = self.density_estimator.fit(x=[permuted_reped_traindata, reped_state_train],
                                                       y=[permuted_reped_traindata],
                                                       epochs=1,
                                                       batch_size=config.batch_size,
                                                       shuffle=True,
                                                       validation_data=([permuted_reped_validdata, reped_state_valid],
                                                  [permuted_reped_validdata]),
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


    def predict(self, test_data, pixelwise=False):
        print('predict start')
        first = 0
        res = []
        while first < test_data.shape[0]:
            last = min(test_data.shape[0], first+config.test_batch_size)
            res.append(self._predict(test_data[first:last], pixelwise))
            first = last
        print('predict finish')
        return np.concatenate(res, axis=0)

    def _predict(self, test_data, pixelwise=False):

        test_size = test_data.shape[0]
        test_data = self._preprocess(test_data)
        all_masks_log_probs = np.zeros([config.num_of_all_masks, test_size])

        for j in range(config.num_of_all_masks):
            stat = j * np.ones([test_size, 1])
            permuted_test_data = self._get_permuted_data(test_data, state)
            made_predict = self.density_estimator.predict([permuted_test_data, state])
            log_probs = -tf.Session().run(logistic_loss(K.constant(permuted_test_data), K.constant(made_predict)))
            all_masks_log_probs[j,:] = log_probs

        res = logsumexp(all_masks_log_probs, axis=0) - np.log(config.num_of_all_masks)
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
