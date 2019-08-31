import rbm.config as config
import time
from sklearn.neural_network import BernoulliRBM as skRBM
from sklearn.utils import check_array, check_random_state, gen_even_slices
from scipy.special import expit
from scipy.misc import logsumexp
import numpy as np
import sys

class MyBernoulliRBM(skRBM):
    def fit(self, X, y=None, reset_attributes=True):
        """Fit the model to the data X.
        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        if reset_attributes:
            self.components_ = np.asarray(
                rng.normal(0, 0.01, (self.n_components, X.shape[1])),
                order='F')
            self.intercept_hidden_ = np.zeros(self.n_components, )
            self.intercept_visible_ = np.zeros(X.shape[1], )
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        verbose = self.verbose
        begin = time.time()
        for iteration in range(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], rng)

            if verbose:
                end = time.time()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                         self.score_samples(X).mean(), end - begin))
                begin = end

        return self


class RBM:
    def __init__(self):
        self.rbm_model = MyBernoulliRBM(n_components = config.hlayer_size, learning_rate = config.learning_rate,
                               batch_size = config.batch_size, n_iter = 1)


    def fit(self, train_data, validation_data):
        print('fit start')
        #self.rbm_model.fit(train_data)
        #print('fit finish')

        ####

        cnt_improved_LL = 0
        best_val_LL = -np.Inf
        for i in range(config.num_of_epochs):
            if i == 0:
                self.rbm_model.fit(train_data)
            else:
                self.rbm_model.fit(train_data, reset_attributes=False)
            current_val_LL = np.sum(self.predict(validation_data))
            if current_val_LL > best_val_LL:
                best_val_LL = current_val_LL
                cnt_improved_LL = 0
                best_attributes = {'components_': self.rbm_model.components_,
                                   'intercept_visible_': self.rbm_model.intercept_visible_,
                                   'intercept_hidden': self.rbm_model.intercept_hidden_}
            else:
                cnt_improved_LL += 1
            if cnt_improved_LL >= config.patience:
                self.rbm_model.components_ = best_attributes['components_']
                self.rbm_model.intercept_visible_ = best_attributes['intercept_visible_']
                self.rbm_model.intercept_hidden_ = best_attributes['intercept_hidden']
                break
        print('fit finish')
        sys.stdout.flush()


    def predict(self, test_data, verbose=False):
        if verbose:
            print('    ps')
        sm = np.zeros([config.density_estimation_MCMC_samples_count, test_data.shape[0]])
        #v = test_data
        v = np.array(np.random.rand(test_data.size).reshape(test_data.shape) < 0.5, dtype=float)
        for i in range(config.density_estimation_MCMC_samples_count):
            if verbose:
                print('    ### ', i)
            for j in range(config.density_estimation_MCMC_burnin):
                v = self.rbm_model.gibbs(v)

            p_h = self.rbm_model.transform(v)
            h = np.random.rand(test_data.shape[0], config.hlayer_size) < p_h
            h = np.array(h, dtype=float)

            p_v = np.dot(h, self.rbm_model.components_)
            p_v += self.rbm_model.intercept_visible_
            p_v = expit(p_v)

            sm[i,:] = np.sum(np.log(p_v) * test_data, axis=1)
            sm[i,:] += np.sum(np.log(1-p_v) * (1 - test_data), axis=1)
        if verbose:
            print('    pf')
        return logsumexp(sm, axis=0) - np.log(config.density_estimation_MCMC_samples_count)

    def generate(self, n):
        pass