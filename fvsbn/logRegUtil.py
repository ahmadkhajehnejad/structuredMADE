from keras.engine.topology import Layer
from keras.callbacks import Callback
from keras import backend as K
import keras.activations as activations
import tensorflow as tf
import numpy as np
import warnings
import fvsbn.config as config
import keras


class MaskedDenseLayer(Layer):

    def __init__(self, output_dim, masks, activation, **kwargs):
        self.output_dim = output_dim
        super(MaskedDenseLayer, self).__init__(**kwargs)
        self._mask = masks
        self._activation = activations.get(activation)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=234),
                                      # 'glorot_uniform',
                                      trainable=True,
                                      dtype="float32")
        self.b_0 = self.add_weight(name='b_0',
                                   shape=(1, self.output_dim),
                                   initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=234),
                                   # 'glorot_uniform',
                                   trainable=True,
                                   dtype="float32")
        super(MaskedDenseLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, l):
        x = l[0]
        state = l[1]

        bs = K.shape(x)[0]
        ks = K.shape(self.kernel)

        tmp_mask = tf.gather(tf.constant(self._mask), K.reshape(state, [-1]))
        masked = tf.multiply(K.tile(K.reshape(self.kernel, [1, ks[0], ks[1]]), [bs, 1, 1]), tmp_mask)
        output = tf.matmul(K.reshape(x, [bs, 1, ks[0]]), masked)
        output = K.reshape(output, [bs, self.output_dim]) + K.tile(self.b_0, [bs, 1])
        return self._activation(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)


class MyEarlyStopping(Callback):
    def __init__(self, model, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto', train_end_epochs=[]):
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

