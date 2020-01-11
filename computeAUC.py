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
    parser.add_argument('-d', '--digit', help='The digit you want to take as anomaly.', action='store', type=int, default=0)
    args = parser.parse_args()

    avg_precision = []
    auprc = []
    digit = args.digit
    with open('saved_data/digit-' + str(digit) + '-test_data.pkl', 'rb') as fin:
            [x_test, y_test] = pickle.load(fin)
    #model = MADE()
    #model.autoencoder.load_weights('saved_models/' + 'digit-' + str(digit) + '.hdf5')
    #pred_log_probs = model.predict(x_test)
    #with open('saved_data/digit-' + str(digit) + '-pred.pkl', 'wb') as fout:
    #    pickle.dump(pred_log_probs, fout)
    with open('saved_data/digit-' + str(digit) + '-pred.pkl', 'rb') as fin:
        pred_log_probs = pickle.load(fin)
    avg_precision.append(average_precision_score(y_test.reshape([-1]), pred_log_probs.reshape([-1])))
    [precisions, recalls, thresholds] = precision_recall_curve(y_test.reshape([-1]), pred_log_probs.reshape([-1]))
    auprc.append( np.sum ( (precisions[:-1] + precisions[1:]) * (recalls[:-1] - recalls[1:]) / 2 ) )
    print(digit, ' avg_pr: ', avg_precision[-1], '   auc: ', auprc[-1])
