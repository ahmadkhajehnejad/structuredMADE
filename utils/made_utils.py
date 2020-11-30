from keras.engine.topology import Layer
from keras.callbacks import Callback
from keras import backend as K
import keras.activations as activations
import tensorflow as tf
import numpy as np
import warnings
import config
import keras


class MaskedDenseLayer(Layer):

    def __init__(self, output_dim, masks ,activation, **kwargs):
        self.output_dim = output_dim
        super(MaskedDenseLayer, self).__init__(**kwargs)
        self._mask = masks
        self._activation = activations.get(activation)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=234), #'glorot_uniform',
                                      trainable=True,
                                      dtype="float32")
        self.b_0 = self.add_weight(name='b_0', 
                                      shape=(1, self.output_dim),
                                      initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=234), #'glorot_uniform',
                                      trainable=True,
                                      dtype="float32")
        super(MaskedDenseLayer, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, l):
        x = l[0]
        state = l[1]

        bs = K.shape(x)[0]
        ks = K.shape(self.kernel)

        # '''
        masks_count = config.num_of_all_masks

        #cond = lambda i, out: tf.less(i, masks_count)
        cond = lambda i, out: i < masks_count

        state2d = K.tile(K.reshape(state, [-1, 1]), [1, ks[0]])
        tmp_zeros = tf.zeros([bs, ks[0]])

        def loop_body(i, output_prev):
            masked_x = tf.where(K.equal(state2d,i), x, tmp_zeros)
            return i + 1, output_prev + tf.matmul(masked_x, tf.multiply( tf.gather(self._mask, i), self.kernel))

        _, matmul_res = tf.while_loop(cond, loop_body, [tf.constant(0), tf.zeros([bs,ks[1]], dtype="float32")], parallel_iterations=1)

        output = matmul_res + K.tile(self.b_0, [bs, 1])
        # '''

 
        #tmp_mask = tf.gather(tf.constant(self._mask), K.reshape(state,[-1]))
        #masked = tf.multiply(K.tile(K.reshape(self.kernel,[1,ks[0],ks[1]]),[bs,1,1]), tmp_mask)
        #output = tf.matmul(K.reshape(x,[bs,1,ks[0]]), masked)
        #output = K.reshape(output,[bs,self.output_dim]) + K.tile(self.b_0, [bs, 1])

        # i = tf.constant(0)
        # cond = lambda i, out: tf.less(i, bs)
        #
        # def loop_body(i, output):
        #     # tmp_mask = tf.reshape( tf.gather(tf.constant(self._mask, dtype="int32"), state[i]), [ks[0]] )
        #     # tmp_mask_bin = tf.mod(tf.bitwise.right_shift(tf.expand_dims(tmp_mask, 1), tf.range(ks[1])), 2)
        #     # masked = tf.multiply(self.kernel, tf.cast( tmp_mask_bin, dtype="float32"))
        #     tmp_mask = tf.gather(tf.constant(self._mask, dtype="float32"), K.reshape(state, [-1])[i])
        #     masked = tf.multiply(self.kernel, tmp_mask)
        #     new_output = tf.reshape(tf.matmul(K.reshape(x[i, :], [1, ks[0]]), masked), [1, -1])
        #     return tf.add(i,1), tf.concat([ tf.gather( output, tf.range(i)), new_output, tf.gather( output, tf.range(i+1, bs))], axis=0)
        #
        # _, output = tf.while_loop(cond, loop_body, [i, tf.zeros([bs,ks[1]], dtype="float32")], parallel_iterations=1)

        return self._activation(output) 

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)


class GraphConvLayer(Layer):

    def __init__(self, num_output_features, adj, activation, aggregation='average', normalize=False, use_bias=True, **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self._num_output_features = num_output_features
        self._adj = tf.constant(adj)
        self._activation = activations.get(activation)
        self._aggregation = aggregation
        self._normalize = normalize
        self._use_bias = use_bias

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        num_input_features = input_shape[-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(num_input_features + self._num_output_features, self._num_output_features),
                                      initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=234),
                                      # 'glorot_uniform',
                                      trainable=True,
                                      dtype="float32")
        if self.use_bias:
            self.b_0 = self.add_weight(name='b_0',
                                       shape=(1, 1, self._num_output_features),
                                       initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=234),
                                       # 'glorot_uniform',
                                       trainable=True,
                                       dtype="float32")
        super(GraphConvLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        bs = K.shape(x)[0]
        num_nodes = K.shape(self._adj)[0]

        if self._aggregation == 'average':
            normalized_adj = tf.divide(self._adj, tf.tile(tf.reduce_sum(self._adj, axis=0, keep_dims=True), [num_nodes, 1]))
            tiled_normalized_adj = tf.tile(tf.expand_dims(normalized_adj, 0), [bs, 1, 1])
            agg_tmp = tf.matmul( tf.transpose(x, [0, 2, 1]), tiled_normalized_adj)
            agg = tf.transpose(agg_tmp, [0, 2, 1])
        else:
            raise NotImplementedError

        tiled_kernel = tf.tile( tf.expand_dims(self.kernel, 0), [bs, 1, 1])
        matmul_res = tf.matmul( tf.concat([x,agg], axis=2), tiled_kernel)

        if self._use_bias:
            output = matmul_res + K.tile(self.b_0, [bs, num_nodes, 1])
        else:
            output = matmul_res

        return self._activation(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self._num_output_features)


def negative_log_likelihood_loss(y_true, y_pred):
    return -K.mean(K.log(y_pred))


class MyEarlyStopping(Callback):
    def __init__(self, model, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto', train_end_epochs = []):
        super(MyEarlyStopping, self).__init__()

        self.model = model
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.train_end_epochs = train_end_epochs
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current - self.min_delta, self.best):
            if self.monitor_op(current, self.best):
                self.best = current
                if config.use_best_validated_weights:
                    self.best_weights = self.model.get_weights()
            self.wait = 0
                
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        '''
        print('           counter:   ', self.wait)
        print('           current:   ', current)
        print('           best:      ', self.best)
        #print('           curr-dlt:  ', current - self.min_delta)
        #print('           min_delta: ', self.min_delta)
        W = self.model.layers[-1].get_weights()[0]
        print('           train_loss_reg: ', logs.get('loss'))
        print('           val_loss_reg: ', logs.get('val_loss'))
        r_ = self.model.reg_coef * K.eval(self.model.reg_function(W))
        print('           train_loss: ', logs.get('loss') - r_, '   ', r_)
        print('           val_loss: ', logs.get('val_loss') - r_, '   ', r_)
        '''


    def on_train_end(self, logs=None):
        self.train_end_epochs.append(self.stopped_epoch)
        if config.use_best_validated_weights:
            self.model.set_weights(self.best_weights)
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))



class MaskedConvLayer(Layer):

    def __init__(self, num_filters, filter_size, activation, include_central_pixel = False, **kwargs):
        if np.mod(filter_size, 2) == 0:
            raise("Filter size should be an odd integer.")
        self.num_filters = num_filters
        self.filter_size = filter_size
        super(MaskedConvLayer, self).__init__(**kwargs)
        self._activation = activations.get(activation)
        mask = np.ones([self.filter_size, self.filter_size])
        mask[ (self.filter_size // 2)+1:,:] = 0
        if include_central_pixel:
            mask[self.filter_size // 2, (self.filter_size // 2) + 1:] = 0
        else:
            mask[self.filter_size // 2, self.filter_size // 2:] = 0
        self.mask = K.constant(mask, dtype="float32")

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.
        self.filters = self.add_weight(name='filters',
                                      shape=( self.filter_size, self.filter_size, input_shape[3], self.num_filters),
                                      initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=234),
                                      # 'glorot_uniform',
                                      trainable=True,
                                      dtype="float32")
        self.b_0 = self.add_weight(name='b_0',
                                   shape=(1, self.num_filters),
                                   initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=234),
                                   # 'glorot_uniform',
                                   trainable=True,
                                   dtype="float32")
        super(MaskedConvLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        num_input_channels = K.shape(inputs)[3]
        reshaped_mask = K.reshape( self.mask, [self.filter_size, self.filter_size, 1, 1])
        tiled_mask = K.tile(reshaped_mask, [ 1, 1, num_input_channels, self.num_filters])
        masked_filters = tf.multiply(self.filters, tiled_mask)
        output = K.conv2d(inputs, masked_filters, padding="same", data_format="channels_last")

        batch_size = K.shape(inputs)[0]
        input_h = K.shape(inputs)[1]
        input_w = K.shape(inputs)[2]
        reshaped_bias = K.reshape( self.b_0, [1, 1, 1, 1, self.num_filters])
        tiled_bias = K.tile(reshaped_bias, [batch_size, input_h, input_w, num_input_channels, 1])
        output += tiled_bias

        return self._activation(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.num_filters)



class StatefulMaskedConvLayer(Layer):

    def __init__(self, num_states, num_filters, filter_size, activation, include_central_pixel = False, **kwargs):
        if np.mod(filter_size, 2) == 0:
            raise("Filter size should be an odd integer.")
        self.num_states = num_states
        self.num_filters = num_filters
        self.filter_size = filter_size
        super(StatefulMaskedConvLayer, self).__init__(**kwargs)
        self._activation = activations.get(activation)
        mask = np.ones([self.filter_size, self.filter_size])
        mask[ (self.filter_size // 2)+1:,:] = 0
        if include_central_pixel:
            mask[self.filter_size // 2, (self.filter_size // 2) + 1:] = 0
        else:
            mask[self.filter_size // 2, self.filter_size // 2:] = 0
        self.mask = K.constant(mask, dtype="float32")

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.filters = self.add_weight(name='filters',
                                      shape=( self.filter_size, self.filter_size, input_shape[0][3], self.num_filters * self.num_states),
                                      initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=234),
                                      # 'glorot_uniform',
                                      trainable=True,
                                      dtype="float32")
        self.b_0 = self.add_weight(name='b_0',
                                   shape=(1, self.num_filters * self.num_states),
                                   initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=234),
                                   # 'glorot_uniform',
                                   trainable=True,
                                   dtype="float32")
        super(StatefulMaskedConvLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, l):
        x = l[0]
        st = l[1]
        num_input_channels = K.shape(x)[3]
        reshaped_mask = K.reshape( self.mask, [self.filter_size, self.filter_size, 1, 1])
        tiled_mask = K.tile(reshaped_mask, [ 1, 1, num_input_channels, self.num_filters * self.num_states])
        masked_filters = tf.multiply(self.filters, tiled_mask)
        output = K.conv2d(x, masked_filters, padding="same", data_format="channels_last")

        batch_size = K.shape(x)[0]
        input_h = K.shape(x)[1]
        input_w = K.shape(x)[2]
        reshaped_bias = K.reshape( self.b_0, [1, 1, 1, self.num_filters * self.num_states])
        tiled_bias = K.tile(reshaped_bias, [batch_size, input_h, input_w, 1])
        output += tiled_bias


        output = tf.reshape(tf.transpose(output, [0,3,1,2]), [-1,input_h, input_w])

        ind = tf.tile( tf.reshape(st, [-1,1]), [1,self.num_filters]) * self.num_filters
        ind = ind + tf.cast(tf.tile(tf.reshape(tf.constant(np.arange(self.num_filters)), [1,-1]), [batch_size, 1]), tf.int32)
        ind = ind + tf.tile(tf.reshape(self.num_filters * self.num_states * tf.range(batch_size), [-1,1]), [1,self.num_filters])

        output = tf.gather(output, tf.reshape(ind, [-1]))
        output = tf.reshape(output, [batch_size, self.num_filters, input_h, input_w])
        output = tf.transpose(output, [0,2,3,1])

        return self._activation(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.num_filters)

