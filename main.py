
import time
import numpy as np
from dataset import get_data
import config
from made import MADE
from patch_made import PatchMADE
import sys
from PIL import Image

import tensorflow as tf
from keras import backend as K

import argparse
import pickle
import os

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
    

def execute_one_round(round_num, all_pi, all_masks):
    
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
    elif args['data_name'] == 'BayesNet':
        args['n'] = config.n_of_BayesNet
        args['par_num'] = config.par_num_of_BayesNet
    elif args['data_name'].startswith('mnist'):
        args['digit'] = config.digit
    
    data = get_data(args)

    print('data loaded')

    # if config.generate_samples:
    #     n = len(data['train_data'])
    #     for i in range(n):
    #         im = Image.fromarray(255*data['train_data'][i,:].reshape([config.height, config.width]))
    #         im.convert('RGB').save(config.generated_samples_dir+'train_' + str(i)+'.png')

    if config.patch_MADE == -1:
        model = MADE(all_pi=all_pi, all_masks=all_masks)
    else:
        model = PatchMADE()

    print('model initiated')
    
    model.fit(data['train_data'], data['valid_data'])
        
    pred = model.predict(data['test_data'])

    res = dict()
    res['NLL'], res['KL'] = evaluate(pred, data['test_data_probs'])
    print('KL: ' + str(res['KL']), file=sys.stderr)
    print('NLL: ' + str(res['NLL']), file=sys.stderr)
    sys.stderr.flush()
    # res['train_end_epochs'] = model.train_end_epochs
    # res['num_of_connections'] = model.num_of_connections()

    if config.generate_samples:
        n = config.num_of_generated_samples_each_execution
        # generated_samples = model.generate(n).reshape(n, config.height, -1, 3)
        generated_samples = model.generate(n).reshape(n, config.height, -1)
        os.makedirs(config.generated_samples_dir, exist_ok=True)
        for i in range(n):
            im = Image.fromarray(256*generated_samples[i,:,:])
            im.convert('RGB').save(config.generated_samples_dir + str(round_num) + '--' + str(i)+'.png')
    return res
    
    

def main(all_pi, all_masks):
    
    print ('algorithm: ', config.algorithm, '\tdata_name:', config.data_name)
    
    NLLs = []
    KLs = []
    num_of_connections = []
    train_end_epochs = []
    start_time = time.time()
    for ne in range(0, config.num_of_exec):
        print('execution #' + str(ne), file=sys.stderr)
        sys.stderr.flush()
        res = execute_one_round(ne, all_pi, all_masks)
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

    parser = argparse.ArgumentParser(description='SMADE')

    parser.add_argument('--make_masks', '-m', action='store_true', help='Just make masks and permutations')
    parser.add_argument('--load_masks', '-l', action='store_true', help='Load masks and permutations')

    args = parser.parse_args()

    if args.make_masks:
        model = MADE()
        with open('masks.pkl','wb') as file:
            pickle.dump([model.all_pi, model.all_masks], file)
    else:
        all_pi, all_masks = None, None
        if args.load_masks:
            with open('masks.pkl', 'rb') as file:
                [all_pi, all_masks] = pickle.load(file)

        main(all_pi, all_masks)

