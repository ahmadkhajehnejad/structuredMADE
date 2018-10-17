
import time
import numpy as np
from dataset import get_data
import config
from made import MADE
import sys    

import tensorflow as tf
from keras import backend as K


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.Session(config=tf_config)
K.set_session(sess)


def evaluate(pred_log_probs, true_probs=None):
    NLL = -1*np.mean(pred_log_probs)
    if true_probs is not None:
        if config.test_size == 'FULL_TEST':
            KL = -1*np.sum(np.multiply(true_probs, (pred_log_probs - np.log(true_probs))))
        else:
            KL = -1*np.mean(pred_log_probs - np.log(true_probs))
        return [NLL, KL]
    return [NLL, None]
    

def execute_one_round():
    
    args = {'data_name' : config.data_name, 'train_size' : config.train_size, 'valid_size' : config.validation_size, 'test_size' : config.test_size}
    if args['data_name'] == 'grid':
        args['width'] = config.width
        args['height'] = config.height
    elif args['data_name'] == 'Boltzmann':
        args['n'] = config.n_boltzmann
        args['m'] = config.m_boltzmann
    elif args['data_name'] == 'k_sparse':
        args['n'] = config.n_of_k_sparse
        args['sparsity_degree'] = config.sparsity_degree
    elif args['data_name'].startswith('mnist'):
        args['digit'] = config.digit
    
    data = get_data(args)

    model = MADE()
    
    model.fit(data['train_data'], data['valid_data'])
        
    pred = model.predict(data['test_data'])
        
    res = dict()    
    res['NLL'], res['KL'] = evaluate(pred, data['test_data_probs'])
    print('KL: ' + str(res['KL']), file=sys.stderr)
    print('NLL: ' + str(res['NLL']), file=sys.stderr)
    sys.stderr.flush()
    res['train_end_epochs'] = model.train_end_epochs
    res['num_of_connections'] = model.num_of_connections()
    return res
    
    

def main():
    
    print ('algorithm: ', config.algorithm, '\tdata_name:', config.data_name)
    
    NLLs = []
    KLs = []
    num_of_connections = []
    train_end_epochs = []
    start_time = time.time()
    for ne in range(0, config.num_of_exec):
        print('execution #' + str(ne), file=sys.stderr)
        sys.stderr.flush()
        res = execute_one_round()
        NLLs.append(res['NLL'])
        KLs.append(res['KL'])
        num_of_connections.append(res['num_of_connections'])
        train_end_epochs.append(res['train_end_epochs'][0])
        
    total_time = time.time() - start_time
    
    print('End Epochs:', train_end_epochs)
    print('End Epochs Average', np.mean(train_end_epochs))
    print('Num of Connections:', num_of_connections)
    print('avg Num of Connections', np.mean(num_of_connections))
    print('NLLs:', NLLs)
    print('Average NLLs:', np.mean(NLLs))
    print('Variance NLLs:', np.var(NLLs))
    if KLs[0] is not None:
        print('KLs:', KLs)
        print('Average KLs:', np.mean(KLs))
        print('Variance KLs:', np.var(KLs))
    print('Total Time:', total_time)

if __name__=='__main__':
    main()

