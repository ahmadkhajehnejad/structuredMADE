from keras.engine.topology import Layer
from keras.callbacks import Callback
from keras import backend as K
import keras.activations as activations
import tensorflow as tf
import numpy as np
import warnings


MDL_masks = []

class MaskedDenseLayer(Layer):
    def __init__(self, output_dim, mask_index ,activation, **kwargs):
        self.output_dim = output_dim
        super(MaskedDenseLayer, self).__init__(**kwargs)
        self._mask_index = mask_index
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
        self.x = l[0]
        self._state = l[1]

        bs = K.shape(self.x)[0]
        ks = K.shape(self.kernel)

        tmp_mask = tf.gather(tf.constant(np.array(MDL_masks[self._mask_index])), K.reshape(self._state,[-1]))
        masked = tf.multiply(K.tile(K.reshape(self.kernel,[1,ks[0],ks[1]]),[bs,1,1]), tmp_mask)
        self._output = tf.matmul(K.reshape(self.x,[bs,1,ks[0]]), masked)
        return self._activation(K.reshape(self._output,[bs,self.output_dim]))
  
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)


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