import mixture_multivariateBernoulli.config as config
import numpy as np
from scipy.misc import logsumexp
import sys


class MMB:

    def _compute_gamma(self, x):
        n = x.shape[0]
        log_gamma = np.zeros([n, config.num_components])
        for j in range(config.num_components):
            b = np.tile(self.mu[j, :].reshape([1, -1]), [n, 1])
            log_gamma[:, j] = np.log(self.pi[j]) + np.sum( (x * np.log(b)) + ((1-x) * np.log(1 - b)), axis=1 )
        log_gamma = log_gamma - np.tile( logsumexp(log_gamma, axis=1).reshape([n,1]), [1,config.num_components])
        return np.exp(log_gamma)

    def _EM_step(self, x):
        d = x.shape[1]
        gamma = self._compute_gamma(x)
        self.pi = np.sum(gamma, axis=0) / np.sum(gamma)
        self.mu = np.matmul( gamma.T, x) / np.tile(np.sum(gamma, axis=0).reshape([-1,1]), [1,d])
        self.mu[self.mu <= 0] = 0.000001
        self.mu[self.mu >= 1] = 0.999999


    def fit(self, train_data, validation_data):
        print('fit start')
        d = train_data.shape[1]
        k = config.num_components

        self.pi = np.random.rand(k)
        self.pi = self.pi / np.sum(self.pi)
        self.mu = np.random.rand(k*d).reshape([k,d])

        best_val_LL = -np.Inf
        best_pi = None
        best_mu = None

        for iter in range(config.num_EMiters):
            print(' EM iter: ', iter)
            self._EM_step(train_data)
            LL = np.mean(self.predict(validation_data))
            is_the_best = False
            if LL > best_val_LL:
                best_val_LL = LL
                best_pi = self.pi.copy()
                best_mu = self.mu.copy()
                is_the_best = True
            print('               val_LL: ', LL, '   is_the_best: ', is_the_best)


        self.pi = best_pi.copy()
        self.mu = best_mu.copy()
        print('fit finish')
        sys.stdout.flush()


    def predict(self, test_data, verbose=False):
        n = test_data.shape[0]
        logp = np.zeros([n, config.num_components])
        for j in range(config.num_components):
            b = np.tile(self.mu[j,:].reshape([1,-1]), [n,1])
            logp[:,j] = np.log(self.pi[j]) + np.sum(test_data * np.log(b) + (1-test_data) * np.log(1-b), axis=1)
        return logsumexp(logp, axis=1)

    def generate(self, n):
        pass