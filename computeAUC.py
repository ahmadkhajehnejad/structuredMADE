import numpy as np
from made import MADE
import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import argparse
import config

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

    avg_precision = []
    auprc = []

    with open('saved_data/digit-' + str(normal_class) + '-test_data.pkl', 'rb') as fin:
            [x_test, y_test] = pickle.load(fin)

    model = MADE()
    model.density_estimator.load_weights('saved_models/' + 'digit-' + str(normal_class) + '.hdf5')
    i_ = 0
    n_ = x_test.shape[0]
    while i_ < n_:
        j_ = min(i_ + config.batch_size, n_)
        tmp = model.predict(x_test[i_:j_,:]).reshape([-1])
        tmp_pixelwise = model.predict(x_test[i_:j_,:], pixelwise=True).reshape([-1, config.height*config.width])
        if i_ == 0:
            pred_log_probs = tmp
            pred_log_probs_pixelwise = tmp_pixelwise
        else:
            pred_log_probs = np.concatenate( [pred_log_probs, tmp], axis=0)
            pred_log_probs_pixelwise = np.concatenate( [pred_log_probs_pixelwise, tmp_pixelwise], axis=0)
        i_ = j_
    with open('saved_data/digit-' + str(normal_class) + '-pred.pkl', 'wb') as fout:
       pickle.dump(pred_log_probs, fout)
    with open('saved_data/digit-' + str(normal_class) + '-pred-pixelwise.pkl', 'wb') as fout:
       pickle.dump(pred_log_probs_pixelwise, fout)
    # with open('saved_data/digit-' + str(normal_class) + '-pred.pkl', 'rb') as fin:
    #     pred_log_probs = pickle.load(fin)

    if data_protocol == 0:
        anomaly_class = normal_class  ## In this protocol, a specific class is assumed as the anomaly class.
        y_binary = np.array(y_test != anomaly_class, dtype=np.int)
    else:
        y_binary = np.array(y_test == normal_class, dtype=np.int)

    if args.curve_type == 0:
        avg_precision.append(average_precision_score(y_binary.reshape([-1]), pred_log_probs.reshape([-1])))
        [precisions, recalls, thresholds] = precision_recall_curve(y_binary.reshape([-1]), pred_log_probs.reshape([-1]))
        auprc.append( np.sum ( (precisions[:-1] + precisions[1:]) * (recalls[:-1] - recalls[1:]) / 2 ) )
        print(normal_class, ' avg_pr: ', avg_precision[-1], '   auc: ', auprc[-1])
    elif args.curve_type == 1:
        auroc = roc_auc_score(y_binary, pred_log_probs.reshape([-1]))
        print(auroc)
    else:
        raise Exception('not implemented curve type ', args.curve_type)
