import numpy as np
from made import MADE
from keras.datasets import mnist
import os
import tensorflow as tf
from keras import backend as K
import pickle
import argparse
import config

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
K.set_session(sess)


def load_cifar_data():

    x_1 = []
    y_1 = []
    for i in range(1, 6):
        with open('datasets/cifar-10-python/cifar-10-batches-py/data_batch_{}'.format(i), 'rb') as fin:
            data_dict = pickle.load(fin, encoding='bytes')
        if i == 1:
            x_1 = data_dict[b'data']
            y_1 = np.array(data_dict[b'labels'])
        else:
            x_1 = np.concatenate([x_1, data_dict[b'data']], axis=0)
            y_1 = np.concatenate([y_1, np.array(data_dict[b'labels'])], axis=0)

    with open('datasets/cifar-10-python/cifar-10-batches-py/test_batch', 'rb') as fin:
        data_dict = pickle.load(fin, encoding='bytes')
    x_2 = data_dict[b'data']
    y_2 = np.array(data_dict[b'labels'])

    return (x_1, y_1) , (x_2, y_2)

def partition_data(x_1, y_1, x_2, y_2, normal_class, data_protocol):
    if data_protocol == 0:
        anomaly_class = normal_class   ## In this protocol, a specific class is assumed as the anomaly class.
        x = np.concatenate([x_1, x_2], axis=0)
        y = np.concatenate([y_1, y_2], axis=0)
        ind_anomaly = np.where(y == anomaly_class)[0]
        ind_normal = np.where(y != anomaly_class)[0]

        num_classes = np.max(y) + 1

        n_train = int(0.8*ind_normal.size)
        ind_train = np.random.choice(ind_normal, n_train, replace=False)
        ind_test_other_classes = list(set(ind_normal).difference(ind_train))
        ind_test = np.concatenate([ind_anomaly, ind_test_other_classes])

        x_train = x[ind_train,:]
        y_train = y[ind_train]
        x_test = x[ind_test,:]
        y_test = y[ind_test]
        # y_test = np.concatenate([np.zeros([len(ind_anomaly)]), np.ones([len(ind_test_other_classes)])])

        n_validation = int(n_train / 10)
        x_validation = x_train[:n_validation,:]
        y_validation = y_train[:n_validation]
        x_train = x_train[n_validation:,:]
        y_train = y_train[n_validation:]

    elif data_protocol == 1:
        inds = np.random.permutation(x_1.shape[0] + x_2.shape[0])
        x = np.concatenate([x_1, x_2], axis=0)[inds]
        y = np.concatenate([y_1, y_2], axis=0)[inds]

        inds_normal = np.where(y == normal_class)[0]
        inds_anomaly = np.where(y != normal_class)[0]

        n_normal_total = inds_normal.size
        n_train_and_validation = int(n_normal_total * 0.8)
        half_n_test = n_normal_total - n_train_and_validation
        n_train = int(n_train_and_validation * 0.8)
        n_validation = n_train_and_validation - n_train

        x_train = x[inds_normal][:n_train]
        y_train = y[inds_normal][:n_train]
        x_validation = x[inds_normal][n_train:n_train + n_validation]
        y_validation = y[inds_normal][n_train:n_train + n_validation]
        x_test = np.concatenate([x[inds_normal][n_train + n_validation:], x[inds_anomaly][:half_n_test]], axis=0)
        y_test = np.concatenate([y[inds_normal][n_train + n_validation:], y[inds_anomaly][:half_n_test]])

    elif data_protocol == 2:
        inds_normal_1 = np.where(y_1 == normal_class)[0]
        x_1 = x_1[inds_normal_1]
        y_1 = y_1[inds_normal_1]
        n_train_and_validation = inds_normal_1.size
        n_train = int(n_train_and_validation * 0.8)
        inds = np.random.permutation(n_train_and_validation)
        x_1 = x_1[inds]
        y_1 = y_1[inds]

        x_train = x_1[:n_train]
        y_train = y_1[:n_train]
        x_validation = x_1[n_train:]
        y_validation = y_1[n_train:]
        x_test = x_2
        y_test = y_2

    return (x_train / 255, y_train), (x_validation / 255, y_validation), (x_test / 255, y_test)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', '--normal_class', help='The class you want to take as anomaly.', action='store', type=int, default=0)
    parser.add_argument('-dp', '--data_protocol', type=int, default=1, help='protocol for train/test partitioning.')
    args = parser.parse_args()

    normal_class = args.normal_class
    data_protocol = args.data_protocol

    if config.data_name.startswith('mnist'):
        (x_1, y_1), (x_2, y_2) = mnist.load_data()
        x_1 = x_1.reshape([x_1.shape[0], -1])
        x_2 = x_2.reshape([x_2.shape[0], -1])
    else:
        (x_1, y_1), (x_2, y_2) = load_cifar_data()
        x_1 = np.transpose( x_1.reshape([-1,3,32,32]), (0,2,3,1))
        x_1 = x_1.reshape([x_1.shape[0], -1])
        x_2 = np.transpose( x_2.reshape([-1,3,32,32]), (0,2,3,1))
        x_2 = x_2.reshape([x_2.shape[0], -1])

    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = partition_data(x_1, y_1, x_2, y_2, normal_class, data_protocol)

    model = MADE()

    print('model initiated')
    model.fit(x_train, x_validation)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model.autoencoder.save_weights('saved_models/' + 'digit-' + str(normal_class) + '.hdf5')


    if not os.path.exists('saved_data'):
        os.makedirs('saved_data')
    with open('saved_data/digit-' + str(normal_class) + '-test_data.pkl', 'wb') as fout:
        pickle.dump([x_test, y_test], fout)


if __name__ == '__main__':
    main()

