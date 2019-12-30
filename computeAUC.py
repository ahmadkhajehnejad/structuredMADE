import numpy as np
from made import MADE
import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score

# tf_config = tf.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# sess = tf.Session(config=tf_config)
# K.set_session(sess)


if __name__ == '__main__':
    auprc = []
    for digit in range(10):
        with open('saved_data/digit-' + str(digit) + '-test_data.pkl', 'rb') as fin:
            [x_test, y_test] = pickle.load(fin)
        model = MADE()
        model.autoencoder.load_weights('saved_models/' + 'digit-' + str(digit) + '.hdf5')
        pred_log_probs = model.predict(x_test)
        auprc.append(precision_recall_curve(y_test.reshape([-1]), pred_log_probs.reshape([-1])))
        print(digit, ': ', auprc[-1])