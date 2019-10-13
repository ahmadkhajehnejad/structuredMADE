import numpy as np
import fvsbn.config as config
from fvsbn.logReg import LogReg
from scipy.misc import logsumexp
import sys

class FVSBN:

    def __init__(self):
        self.logreg = [None] * config.number_of_permutations

    def fit(self, train_data, validation_data):
        print('FVSBN fit start')

        for j in range(config.number_of_permutations):
            self.logreg[j] = LogReg()
            self.logreg[j].fit(train_data, validation_data)

        print('FVSBN fit finish')
        sys.stdout.flush()

    def predict(self, test_data):
        n = test_data.shape[0]
        log_probs = np.zeros([n, config.number_of_permutations])
        for j in range(config.number_of_permutations):
            log_probs[:, j] += self.logreg[j].predict(test_data)
        return logsumexp(log_probs, axis=1) - np.log(config.number_of_permutations)

    def generate(self, n):
        pass