from keras.engine.topology import Layer
from keras.callbacks import Callback
from keras import backend as K
import keras.activations as activations
import tensorflow as tf
import numpy as np
import warnings


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
                                      initializer='glorot_uniform',
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
        return self._activation(K.reshape(output,[bs,self.output_dim]))
  
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)


class MaskedDenseLayerMultiMasks(Layer):
    def __init__(self, output_dim, masks, activation, **kwargs):
        self.output_dim = output_dim
        super(MaskedDenseLayerMultiMasks, self).__init__(**kwargs)
        self._masks = masks
        self._activation = activations.get(activation)
        self.num_masks = self._masks.shape[0]

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=False,
                                      dtype="float32")

        self.masked_kernel = tf.multiply(
            K.tile(K.reshape(self.kernel, [1, input_shape[2], self.output_dim]),
                   [self.num_masks, 1, 1]
                   ), self._masks
        )
        super(MaskedDenseLayerMultiMasks, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        #bs = K.shape(x)[-3]

        #print_node = tf.Print(bs, [tf.constant(1111), bs])

        '''
        masked = K.tile(\
            K.reshape(\
                tf.multiply(\
                    K.tile(K.reshape(self.kernel, [1, ks[0], ks[1]]),\
                           [self.num_masks, 1, 1]\
                           ), self._masks\
                    ), [1, self.num_masks, ks[0], ks[1]]),
            [bs, 1, 1, 1])
            #[print_node, 1, 1, 1])
        output = tf.matmul(K.reshape(x, [bs, self.num_masks, 1, ks[0]]), masked)
        return self._activation(K.reshape(output, [bs, self.num_masks, self.output_dim]))
        '''

        output = tf.scan(self._masked_transform, x, initializer=tf.zeros([self.num_masks, K.shape(self.kernel)[1]]))

        return output

    def _masked_transform(self, _nothing, x):
        ks = K.shape(self.kernel)
        return K.reshape(tf.matmul(K.reshape(x, [self.num_masks, 1, ks[0]]), self.masked_kernel), [self.num_masks, ks[1]])


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_masks, self.output_dim)


class LikelihoodConvexCombinationLayer(Layer):
    def __init__(self, num_models, **kwargs):
        super(LikelihoodConvexCombinationLayer, self).__init__(**kwargs)
        self.num_models = num_models

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.alpha = self.add_weight(name='alpha',
                                      shape=(self.num_models,),
                                      initializer='Zeros',
                                      trainable=True,
                                      dtype="float32")
        super(LikelihoodConvexCombinationLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input):
        y, x = input

        bs = K.shape(y)[0]

        likelihoods = K.prod(tf.multiply(tf.pow(y,x), tf.pow(1-y, 1-x)), axis=2)
        exp_alpha = K.reshape(K.exp(self.alpha), [self.num_models, 1])
        return tf.reshape(tf.matmul(likelihoods, exp_alpha), [bs, 1]) / K.sum(exp_alpha)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def negative_log_likelihood_loss(y_true, y_pred):
    return -K.mean(K.log(y_pred))


class MyEarlyStopping(Callback):
    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto', train_end_epochs = []):
        super(MyEarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.train_end_epochs = train_end_epochs

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
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.train_end_epochs.append(self.stopped_epoch)
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


def log_sum_exp(a, axis=0, keepdims=False):
    mx = np.max( a, axis = axis, keepdims=keepdims)
    tile_shape = np.ones([len(a.shape),], dtype=int)
    tile_shape[axis] = a.shape[axis]
    tmp_shape = [i for i in a.shape]
    tmp_shape[axis] = 1
    res = mx + np.log(np.sum( np.exp(a-np.tile(mx.reshape(tmp_shape),tile_shape)), axis=axis, keepdims=keepdims))
    return res