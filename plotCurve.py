import numpy as np
import pickle
from sklearn.metrics import roc_curve
import argparse
import matplotlib.pyplot as plt

# tf_config = tf.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# sess = tf.Session(config=tf_config)
# K.set_session(sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', '--normal_class', help='The digit you want to take as anomaly.', action='store', type=int, default=0)
    parser.add_argument('-dp', '--data_protocol', type=int, default=1, help='protocol for train/test partitioning.')
    parser.add_argument('-cvt', '--curve_type', type=int, default=0, help='Curve type: 0 for AUPRC and 1 for ROC-AUC.')
    args = parser.parse_args()

    data_protocol = args.data_protocol
    normal_class = args.normal_class

    with open('saved_data/digit-' + str(normal_class) + '-test_data-a.pkl', 'rb') as fin:
        [ _, y_test_a] = pickle.load(fin)

    with open('saved_data/digit-' + str(normal_class) + '-test_data-b.pkl', 'rb') as fin:
        [ _, y_test_b] = pickle.load(fin)

    with open('saved_data/digit-' + str(normal_class) + '-pred-a.pkl', 'rb') as fin:
        pred_log_probs_a = pickle.load(fin)
    with open('saved_data/digit-' + str(normal_class) + '-pred-b.pkl', 'rb') as fin:
        pred_log_probs_b = pickle.load(fin)

    if data_protocol == 0:
        anomaly_class = normal_class  ## In this protocol, a specific class is assumed as the anomaly class.
        y_binary_a = np.array(y_test_a != anomaly_class, dtype=np.int)
        y_binary_b = np.array(y_test_b != anomaly_class, dtype=np.int)
    else:
        y_binary_a = np.array(y_test_a == normal_class, dtype=np.int)
        y_binary_b = np.array(y_test_b == normal_class, dtype=np.int)

    if args.curve_type == 0:
        raise Exception('not implemented')
    elif args.curve_type == 1:
        fpr_a, tpr_a, _ = roc_curve(y_binary_a.reshape([-1]), pred_log_probs_a.reshape([-1]))
        fpr_b, tpr_b, _ = roc_curve(y_binary_b.reshape([-1]), pred_log_probs_b.reshape([-1]))

        plt.figure()
        plt.plot(fpr_a, tpr_a, color='darkorange', lw=2, label='a')
        plt.plot(fpr_b, tpr_b, color='blue', lw=b, label='a')
        plt.plot([0, 1], [0, 1], color='cyan', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
    else:
        raise Exception('not implemented curve type ', args.curve_type)
