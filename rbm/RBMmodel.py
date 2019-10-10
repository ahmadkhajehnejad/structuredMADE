import rbm.config as config
import numpy as np
import rbm.pydeep.rbm.model as model
import rbm.pydeep.rbm.trainer as trainer
import rbm.pydeep.rbm.estimator as estimator
import sys


class RBM:
    def __init__(self, train_data):
        self.update_offsets = 0.01
        if self.update_offsets <= 0.0:
            self.rbm = model.BinaryBinaryRBM(number_visibles=config.graph_size,
                                        number_hiddens=config.hlayer_size,
                                        data=None,
                                        initial_weights=0.01,
                                        initial_visible_bias=0.0,
                                        initial_hidden_bias=0.0,
                                        initial_visible_offsets=0.0,
                                        initial_hidden_offsets=0.0)
        else:
            self.rbm = model.BinaryBinaryRBM(number_visibles=config.graph_size,
                                        number_hiddens=config.hlayer_size,
                                        data=train_data,
                                        initial_weights=0.01,
                                        initial_visible_bias='AUTO',
                                        initial_hidden_bias='AUTO',
                                        initial_visible_offsets='AUTO',
                                        initial_hidden_offsets='AUTO')

        self.trainer_pcd = trainer.PCD(self.rbm, num_chains=config.batch_size)
        #self.rbm_model = MyBernoulliRBM(n_components = config.hlayer_size, learning_rate = config.learning_rate,
        #                       batch_size = config.batch_size, n_iter = 1)


    def fit(self, train_data, validation_data):
        print('fit start')

        # Train model
        print('Training')
        print('Epoch\t\tRecon. Error\tLog likelihood \tExpected End-Time')
        best_val_LL = -np.Inf
        early_stop_cnt = 0
        for epoch in range(1, config.num_of_epochs + 1):

            # Loop over all batches
            for b in range(0, train_data.shape[0], config.batch_size):
                batch = train_data[b:b + config.batch_size, :]
                self.trainer_pcd.train(data=batch,
                                  epsilon=0.01,
                                  update_visible_offsets=self.update_offsets,
                                  update_hidden_offsets=self.update_offsets)

            val_LL = np.mean(self.predict(validation_data))
            if val_LL > best_val_LL:
                best_val_LL = val_LL
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if early_stop_cnt > config.patience:
                break

            print(epoch, '         best_val_LL:', best_val_LL, '     early_stop_cnt:', early_stop_cnt)

            # Calculate reconstruction error and expected end time every 10th epoch
            #if epoch % 10 == 0:
            #    RE = np.mean(estimator.reconstruction_error(self.rbm, train_data))
            #    # print('{}\t\t{:.4f}\t\t\t{}'.format(
            #    #     epoch, RE))
            #else:
            #    print(epoch, '         best_val_LL', best_val_LL)

        ####

        print('fit finish')
        sys.stdout.flush()


    def predict(self, test_data, verbose=False):
        logZ_approx_AIS = estimator.annealed_importance_sampling(self.rbm)[0]
        return np.mean(estimator.log_likelihood_v(self.rbm, logZ_approx_AIS, test_data))

    def generate(self, n):
        pass