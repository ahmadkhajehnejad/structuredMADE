import config
import numpy as np
from utils import grid_orders
from utils.made_utils import MaskedDenseLayer
from keras.layers import Input, Concatenate
from keras.models import Model
import keras.backend as K
from scipy.special import logsumexp

# MIN_VAR = 0.0001

class PatchMADE:

    def __init__(self):
        self.masking_method = config.algorithm
        if self.masking_method != 'orig':
            raise Exception('Now just orig algorithm is supported')

        self.subgraph_size = config.width * config.patch_MADE
        self.all_masks, self.all_pi = self.generate_all_masks()

        self.density_estimator = self.build_autoencoder()

    def generate_all_masks(self):
        all_masks = []
        all_pi = []

        for i_m in range(0, config.num_of_all_masks):

            if config.random_dimensions_order == False:
                pi = np.arange(config.graph_size)
            elif config.random_dimensions_order == 'grid':
                pi = grid_orders.get_random_order(config.width, config.height)
            else:
                raise Exception('Error - random_dimensions_order: ' + str(config.random_dimensions_order))
            all_pi.append(pi)

            all_masks.append(self._normal_mask(self.masking_method))

        swapped_all_masks = []
        for i in range(config.num_of_hlayer + 1):
            swapped_masks = []
            for j in range(config.num_of_all_masks):
                swapped_masks.append(all_masks[j][i])
            swapped_all_masks.append(swapped_masks)

        return swapped_all_masks, all_pi

    def _normal_mask(self, masking_method):

        if masking_method != 'orig':
            raise Exception("wrong masking method " + masking_method)

        labels = np.zeros([config.num_of_hlayer, config.hlayer_size], dtype=int)
        min_label = self.subgraph_size
        for ii in range(config.num_of_hlayer):
            labels[ii][:] = np.random.randint(min_label, 2*self.subgraph_size, (config.hlayer_size))
            min_label = np.amin(labels[ii])

        masks = []

        # first layer mask
        mask = np.zeros([2*self.subgraph_size, config.hlayer_size], dtype=np.float32)
        for j in range(0, config.hlayer_size):
            for k in range(0, self.subgraph_size):
                if (labels[0][j] >= k):
                    mask[k][j] = 1.0
        masks.append(mask)

        # hidden layers mask
        for i in range(1, config.num_of_hlayer):
            mask = np.zeros([config.hlayer_size, config.hlayer_size], dtype=np.float32)
            for j in range(0, config.hlayer_size):
                for k in range(0, config.hlayer_size):
                    if (labels[i][j] >= labels[i - 1][k]):
                        mask[k][j] = 1.0
            masks.append(mask)

        # last layer mask
        mask = np.zeros([config.hlayer_size, self.subgraph_size], dtype=np.float32)
        for j in range(0, self.subgraph_size):
            for k in range(0, config.hlayer_size):
                if (self.subgraph_size + j > labels[-1][k]):
                    mask[k][j] = 1.0

        if config.direct_links != False:
            tmp_mask = np.zeros([2*self.subgraph_size, self.subgraph_size], dtype=np.float32)
            for j in range(0, self.subgraph_size):
                if (config.direct_links == True) or (config.direct_links == 'Full'):
                    tmp_mask[:(self.subgraph_size + j), j] = 1.0
                else:
                    raise Exception('Error' + str(config.direct_links))

            # print(tmp_mask.shape)
            mask = np.concatenate([mask, tmp_mask], axis=0)
            # mask = tmp_mask

        masks.append(mask)
        return masks

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
            final_layer_mu.append(MaskedDenseLayer(self.subgraph_size, np.array(self.all_masks[-1]), 'sigmoid'))
            final_layer_logVar.append(MaskedDenseLayer(self.subgraph_size, np.array(self.all_masks[-1]), 'linear'))
            final_layer_logpi_unnormalized.append(
                MaskedDenseLayer(self.subgraph_size, np.array(self.all_masks[-1]), 'linear'))

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

        input_layer = Input(shape=(2*self.subgraph_size,))
        state = Input(shape=(1,), dtype="int32")

        density_estimator = Model(inputs=[input_layer, state], outputs=[get_autoencode(input_layer, state)])

        def normal_loss(y_true, y_pred):
            tmp_sz = config.num_mixture_components * self.subgraph_size
            mu_pred, logVar_pred, logpi_unnormalized_pred = y_pred[:, :tmp_sz], y_pred[:, tmp_sz:2*tmp_sz], y_pred[:, 2*tmp_sz:]

            mu_pred = K.reshape(mu_pred, [-1, config.num_mixture_components, self.subgraph_size])
            logVar_pred = K.reshape(logVar_pred, [-1, config.num_mixture_components, self.subgraph_size])
            logpi_unnormalized_pred = K.reshape(logpi_unnormalized_pred, [-1, config.num_mixture_components, self.subgraph_size])
            logpi_pred = logpi_unnormalized_pred - K.tile(K.logsumexp(logpi_unnormalized_pred, axis=1, keepdims=True), [1, config.num_mixture_components, 1])

            #### 0-255
            #logVar_pred = K.log(K.exp(logVar_pred) + MIN_VAR)

            y_true_tiled = K.tile(K.expand_dims(y_true, 1), [1, config.num_mixture_components, 1])

            tmp = logpi_pred - 0.5 * logVar_pred - 0.5 * np.log(2*np.pi) - 0.5 * K.pow(y_true_tiled - mu_pred, 2) / K.exp(logVar_pred)

            tmp = K.logsumexp(tmp, axis=1)
            tmp = tmp - np.log(256)
            tmp = -K.sum(tmp, axis=1)

            return tmp

        self.loss_function = normal_loss

        density_estimator.compile(optimizer=config.optimizer, loss=normal_loss)
        return density_estimator


    def _get_patched_data(self, data, state):
        n = data.shape[0]
        data_permuted = np.zeros(data.shape)
        for i in range(n):
            data_permuted[i, :] = data[i, np.argsort(self.all_pi[state[i]])]
        tmp = data_permuted.reshape([n, -1, self.subgraph_size])
        zeros = np.zeros([n, 1, self.subgraph_size])
        tmp = np.concatenate([zeros, tmp], axis=1)
        l = [np.concatenate([tmp[:, i, :], tmp[:, i+1, :]], axis=1) for i in range(tmp.shape[1] - 1)]
        return np.concatenate(l, axis=0), np.tile(state, [tmp.shape[1]-1])


    def fit(self, train_data_clean, validation_data_clean):
        cnt = 0
        best_loss = np.Inf
        best_weights = None
        for i in range(config.num_of_epochs):
            if config.use_uniform_noise_for_pmf:
                train_data = train_data_clean + np.random.rand(np.prod(train_data_clean.shape)).reshape(train_data_clean.shape) / 256
                validation_data = validation_data_clean + np.random.rand(np.prod(validation_data_clean.shape)).reshape(validation_data_clean.shape) / 256
            else:
                train_data = train_data_clean
                validation_data = validation_data_clean

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

            patched_reped_train_data, patched_reped_state_train = self._get_patched_data(reped_traindata, reped_state_train)
            patched_reped_valid_data, patched_reped_state_valid = self._get_patched_data(reped_validdata, reped_state_valid)

            verbose = 1
            if verbose > 0:
                print('## Epoch:', i)
            train_history = self.density_estimator.fit(x=[patched_reped_train_data, patched_reped_state_train],
                                                       y=[patched_reped_train_data[:, self.subgraph_size:]],
                                                       epochs=1,
                                                       batch_size=config.batch_size,
                                                       shuffle=True,
                                                       validation_data=([patched_reped_valid_data, patched_reped_state_valid],
                                                  [patched_reped_valid_data[:, self.subgraph_size:]]),
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
        first = 0
        res = []
        while first < test_data.shape[0]:
            last = min(test_data.shape, first+config.batch_size)
            res.append(self._predict(test_data[first:last], pixelwise))
            first = last
        return np.concatenate(res, axis=0)


    def _predict(self, test_data, pixelwise=False):

        if config.use_uniform_noise_for_pmf:
            test_data = test_data + np.random.rand(np.prod(test_data.shape)).reshape(test_data.shape) / 256

        model = self.density_estimator

        print('predict start')
        test_size = test_data.shape[0]
        if not pixelwise:
            all_masks_log_probs = np.zeros([config.num_of_all_masks, test_size])
        else:
            all_masks_log_probs = np.zeros([config.num_of_all_masks, test_size, config.graph_size])

        for j in range(config.num_of_all_masks):

            patched_test_data, st = self._get_patched_data(test_data, j*np.ones([test_data.shape[0]]))

            made_predict = model.predict([patched_test_data, st])

            tmp_sz = config.num_mixture_components * self.subgraph_size
            made_predict_mu = made_predict[ :, :tmp_sz]
            made_predict_logVar = made_predict[ :, tmp_sz:2*tmp_sz]
            made_predict_logpi_unnormalized = made_predict[ :, 2*tmp_sz:3*tmp_sz]

            made_predict_mu = np.reshape(made_predict_mu, [-1, config.num_mixture_components, self.subgraph_size])
            made_predict_logVar = np.reshape(made_predict_logVar, [-1, config.num_mixture_components, self.subgraph_size])
            made_predict_logpi_unnormalized = np.reshape(made_predict_logpi_unnormalized, [-1, config.num_mixture_components, self.subgraph_size])
            made_predict_logpi = made_predict_logpi_unnormalized - np.tile(logsumexp(made_predict_logpi_unnormalized, axis=1, keepdims=True), [1, config.num_mixture_components, 1])

            #made_predict_logVar = np.log(np.exp(made_predict_logVar) + MIN_VAR)

            test_data_tiled = np.tile(np.expand_dims(test_data, 1), [1, config.num_mixture_components, 1])

            tmp = -0.5 * (test_data_tiled - made_predict_mu)**2 / np.exp(made_predict_logVar) - made_predict_logVar/2 - np.log(2*np.pi)/2 + made_predict_logpi

            log_probs = logsumexp(tmp, axis=1)

            log_probs = log_probs - np.log(256)

            log_probs = log_probs.reshape([-1, test_size, self.subgraph_size])
            log_probs = log_probs.transpose([1, 0, 2]).reshape([test_size, -1])
            log_probs = log_probs[:, self.all_pi[j]]

            if not pixelwise:
                made_log_prob = np.sum(log_probs, axis=1)
                all_masks_log_probs[j][:] = made_log_prob
            else:
                all_masks_log_probs[j,:,:] = log_probs


        #res = np.log(np.mean(probs, axis=0))
        res = logsumexp(all_masks_log_probs, axis=0) - np.log(config.num_of_all_masks)
        print('predict finish')
        return res

    def generate(self, n):
        model = self.density_estimator
        k = 0
        tmp = np.zeros([n, self.subgraph_size])
        generated_samples = np.zeros([n, config.graph_size])

        st = np.random.randint(0,config.num_of_all_masks,n)

        while (k+1) * (self.subgraph_size) < config.graph_size:
            inp = np.zeros([n, 2*self.subgraph_size])
            inp[:, :self.subgraph_size] = tmp
            for i in range(self.subgraph_size):
                pred = model.predict([inp, st])
                mu = pred[:, :self.subgraph_size][:, i]
                # logVar = np.log(np.exp(pred[ :, self.subgraph_size:2*self.subgraph_size][:,i]) + MIN_VAR)
                logVar = pred[:, self.subgraph_size:2 * self.subgraph_size][:, i]
                inp[i] = np.random.normal(mu, np.exp(logVar / 2))
            tmp = inp
            generated_samples[:, (k*(self.subgraph_size)) : ((k+1)*(self.subgraph_size))] = tmp

        for i in range(n):
            generated_samples[i,:] = generated_samples[i, self.all_pi[st[i]]]
