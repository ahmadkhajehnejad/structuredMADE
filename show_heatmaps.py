import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import argparse
import config
import matplotlib.pyplot as plt

# tf_config = tf.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# sess = tf.Session(config=tf_config)
# K.set_session(sess)

NUM_IMAGES = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', '--normal_class', help='The digit you want to take as anomaly.', action='store', type=int, default=0)
    parser.add_argument('-dp', '--data_protocol', type=int, default=1, help='protocol for train/test partitioning.')
    args = parser.parse_args()

    normal_class = args.normal_class
    data_protocol = args.data_protocol

    with open('saved_data/digit-' + str(normal_class) + '-test_data.pkl', 'rb') as fin:
            [x_test, y_test] = pickle.load(fin)


    with open('saved_data/digit-' + str(normal_class) + '-pred.pkl', 'rb') as fin:
        pred_log_probs = pickle.load(fin)
    with open('saved_data/digit-' + str(normal_class) + '-pred-pixelwise.pkl', 'rb') as fin:
        pred_log_probs_pixelwise = pickle.load(fin).reshape([-1, config.height*config.width])

    if data_protocol == 0:
        anomaly_class = normal_class  ## In this protocol, a specific class is assumed as the anomaly class.
        y_binary = np.array(y_test != anomaly_class, dtype=np.int)
    else:
        y_binary = np.array(y_test == normal_class, dtype=np.int)

    ####################

    print(y_binary.shape, '   ', pred_log_probs.shape, '   ', pred_log_probs_pixelwise.shape, '   ', x_test.shape)

    ind_normal = y_binary == 1
    normal_pred_log_probs = pred_log_probs[ind_normal]
    normal_pred_log_probs_pixelwise = pred_log_probs_pixelwise[ind_normal,:]
    normal_x_test = x_test[ind_normal,:]

    ind = np.argsort(normal_pred_log_probs)[:NUM_IMAGES]

    plt.figure()
    for i in range(NUM_IMAGES):
        plt.subplot(NUM_IMAGES, 2, 2*i+1)
        img = np.exp(normal_pred_log_probs_pixelwise[ind[i],:].reshape([config.height, config.width]))
        img = (np.round( 255 * (img / np.max(img)) ) ).astype(int)
        plt.imshow(img, cmap='gray')

        plt.subplot(NUM_IMAGES, 2, 2*i+2)
        img = normal_x_test[ind[i],:].reshape([config.height, config.width])
        img = (np.round( 255 * img)).astype(int)
        plt.imshow(img, cmap='gray')

    plt.savefig('./false_negative-'+str(normal_class)+'.png')

    ####################

    ind_anomaly = y_binary == 0
    anomaly_pred_log_probs = pred_log_probs[ind_anomaly]
    anomaly_pred_log_probs_pixelwise = pred_log_probs_pixelwise[ind_anomaly, :]
    anomaly_x_test = x_test[ind_anomaly, :]

    ind = np.argsort(anomaly_pred_log_probs)[-1:-(NUM_IMAGES+1):-1]

    plt.figure()
    for i in range(NUM_IMAGES):
        plt.subplot(NUM_IMAGES, 2, 2*i+1)
        img = np.exp(anomaly_pred_log_probs_pixelwise[ind[i], :].reshape([config.height, config.width]))
        img = (np.round(255 * (img / np.max(img)))).astype(int)
        plt.imshow(img, cmap='gray')

        plt.subplot(NUM_IMAGES, 2, 2*i+2)
        img = anomaly_x_test[ind[i], :].reshape([config.height, config.width])
        img = (np.round(255 * img)).astype(int)
        plt.imshow(img, cmap='gray')

    plt.savefig('./false_positive-'+str(normal_class)+'.png')
