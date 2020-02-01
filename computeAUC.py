import numpy as np
from made import MADE
import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score
import argparse

# tf_config = tf.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# sess = tf.Session(config=tf_config)
# K.set_session(sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', '--normal_class', help='The digit you want to take as anomaly.', action='store', type=int, default=0)
    parser.add_argument('-dp', '--data_protocol', type=int, default=1, help='protocol for train/test partitioning.')
    args = parser.parse_args()

    data_protocol = args.data_protocol
    normal_class = args.normal_class

    avg_precision = []
    auprc = []

    with open('saved_data/digit-' + str(normal_class) + '-test_data.pkl', 'rb') as fin:
            [x_test, y_test] = pickle.load(fin)

    model = MADE()
    model.autoencoder.load_weights('saved_models/' + 'digit-' + str(normal_class) + '.hdf5')
    pred_log_probs = model.predict(x_test)
    with open('saved_data/digit-' + str(normal_class) + '-pred.pkl', 'wb') as fout:
       pickle.dump(pred_log_probs, fout)
    # with open('saved_data/digit-' + str(normal_class) + '-pred.pkl', 'rb') as fin:
    #     pred_log_probs = pickle.load(fin)

    if data_protocol == 0:
        anomaly_class = normal_class  ## In this protocol, a specific class is assumed as the anomaly class.
        y_binary = np.array(y_test != anomaly_class, dtype=np.int)
    else:
        y_binary = np.array(y_test == normal_class, dtype=np.int)

    avg_precision.append(average_precision_score(y_binary.reshape([-1]), pred_log_probs.reshape([-1])))
    [precisions, recalls, thresholds] = precision_recall_curve(y_binary.reshape([-1]), pred_log_probs.reshape([-1]))
    auprc.append( np.sum ( (precisions[:-1] + precisions[1:]) * (recalls[:-1] - recalls[1:]) / 2 ) )
    print(normal_class, ' avg_pr: ', avg_precision[-1], '   auc: ', auprc[-1])
