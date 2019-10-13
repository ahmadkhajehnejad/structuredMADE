import numpy as np
from keras.models import Model
from keras.layers import Input
import fvsbn.config as config
from fvsbn.logRegUtil import MaskedDenseLayer, MyEarlyStopping



class LogReg:

    def __init__(self):
        self.mask, self.pi = self.generate_all_masks()
        self.autoencoder = self.build_autoencoder()
        self.train_end_epochs = []

    def generate_all_masks(self):
        pi = np.random.permutation(config.graph_size)
        mask = self._normal_mask('orig', pi)

        return mask, pi

    def _normal_mask(self, masking_method, pi):
        masks = []
        mask = np.zeros([config.graph_size, config.graph_size], dtype=np.float32)
        for j in range(0, config.graph_size):
            mask[pi < pi[j], j] = 1.0

        masks.append(mask)
        return masks


    def build_autoencoder(self):
        input_layer = Input(shape=(config.graph_size,))
        state = Input(shape=(1,), dtype="int32")
        output_layer = MaskedDenseLayer(config.graph_size, np.array(self.mask), 'sigmoid')([input_layer, state])
        autoencoder = Model(inputs=[input_layer, state], outputs=[output_layer])
        autoencoder.compile(optimizer=config.optimizer, loss='binary_crossentropy')
        return autoencoder


    def fit(self, train_data, validation_data):
        print('      logreg fit start')
        early_stop = MyEarlyStopping(self.autoencoder, monitor='val_loss', min_delta=-0.0, patience=config.patience,
                                     verbose=0, mode='auto',
                                     train_end_epochs=self.train_end_epochs)
        train_size = train_data.shape[0]
        state_train = np.zeros([train_size,1]).astype(np.int32)
        validation_size = validation_data.shape[0]
        state_validation = np.zeros([validation_size,1]).astype(np.int32)
        for i in range(0, config.fit_iter):
            self.autoencoder.fit(x=[train_data, state_train],
                                 y=[train_data],
                                 epochs=config.num_of_epochs,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 validation_data=([validation_data, state_validation],
                                                  [validation_data]),
                                 callbacks=[early_stop],
                                 verbose=0)
        print('      logreg fit finish')


    def predict(self, test_data):
        print('      logreg predict start')
        test_size = test_data.shape[0]

        made_predict = self.autoencoder.predict([test_data, np.zeros([test_size, 1])])
        eps = 0.00001
        made_predict[made_predict < eps] = eps
        made_predict[made_predict > 1 - eps] = 1 - eps

        corrected_log_probs = (np.log(made_predict) * test_data) + (np.log(1 - made_predict) * (1 - test_data))
        made_log_prob = np.sum(corrected_log_probs, axis=1)

        print('      logreg predict finish')
        return made_log_prob
