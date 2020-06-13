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

        tmp_mask = tf.gather(tf.constant(self._mask), K.reshape(state,[-1]))
        masked = tf.multiply(K.tile(K.reshape(self.kernel,[1,ks[0],ks[1]]),[bs,1,1]), tmp_mask)
        output = tf.matmul(K.reshape(x,[bs,1,ks[0]]), masked)
        output = K.reshape(output,[bs,self.output_dim]) + K.tile(self.b_0, [bs, 1])

        return self._activation(output) 

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)


class MaskedConvLayer(Layer):

    def __init__(self, num_filters, filter_size, activation, **kwargs):
        if np.mod(filter_size, 2) == 0:
            raise("Filter size should be an odd integer.")
        self.num_filters = num_filters
        self.filter_size = filter_size
        super(MaskedConvLayer, self).__init__(**kwargs)
        self._activation = activations.get(activation)
        mask = np.zeros([self.filter_size, self.filter_size])
        mask[ (self.filter_size // 2)+1:,:] = 0
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
        reshaped_bias = K.reshape( self.b_0, [1, 1, 1, self.num_filters])
        tiled_bias = K.tile(reshaped_bias, [batch_size, input_h, input_w, 1])
        output += tiled_bias

        return self._activation(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.num_filters)
