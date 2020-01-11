import numpy as np
from made import MADE
from keras.datasets import mnist
import os
import tensorflow as tf
from keras import backend as K
import pickle
import argparse

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
K.set_session(sess)


def test_for_digit(x, y, digit):

    ind_digit = np.where(y == digit)[0]
    ind_other_digits = np.where(y != digit)[0]

    n_train = 9*5600
    ind_train = np.random.choice(ind_other_digits, n_train, replace=False)
    ind_test_other_digits = list(set(ind_other_digits).difference(ind_train))
    ind_test = np.concatenate([ind_digit, ind_test_other_digits])

    x_train = x[ind_train,:]
    x_test = x[ind_test,:]
    y_test = np.concatenate([np.zeros([len(ind_digit)]), np.ones([len(ind_test_other_digits)])])

    n_validation = int(n_train / 10)
    x_validation = x_train[:n_validation,:]
    x_train = x_train[n_validation:,:]

    model = MADE()

    print('model initiated')

    model.fit(x_train, x_validation)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model.autoencoder.save_weights('saved_models/'+'digit-'+ str(digit) + '.hdf5')

    if not os.path.exists('saved_data'):
        os.makedirs('saved_data')
    with open('saved_data/digit-'+str(digit)+'-test_data.pkl', 'wb') as fout:
        pickle.dump([x_test, y_test], fout)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--digit', help='The digit you want to take as anomaly.', action='store', type=int, default=0)
    args = parser.parse_args()


    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    x = x.reshape([x.shape[0],-1])/255.0

    np.random.shuffle(x)
    np.random.shuffle(y)

    test_for_digit(x.copy(), y.copy(), args.digit)



if __name__ == '__main__':
    main()

