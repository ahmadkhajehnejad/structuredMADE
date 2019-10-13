import numpy as np
from sklearn.linear_model import LogisticRegression
import fvsbn.config as config
from scipy.misc import logsumexp
import sys
import copy

class FVSBN:

    def __init__(self):
        self.mu_0 = [None] * config.number_of_permutations
        self.clf = [None] * config.number_of_permutations
        self.pi = [None] * config.number_of_permutations
        self.unique_label = [None] * config.number_of_permutations

    def fit(self, train_data, validation_data):
        print('fit start')
        d = train_data.shape[1]

        for j in range(config.number_of_permutations):
            self.pi[j] = np.random.permutation(d)
            self.unique_label[j] = [None] * d
            if np.unique(train_data[:,self.pi[j][0]]).size == 1:
                self.unique_label[j][0] = train_data[0,self.pi[j][0]]
            else:
                self.mu_0[j] = np.sum(train_data[:, self.pi[j][0]] == 1) / train_data.shape[0]
            self.clf[j] = [None] * d
            for i in range(1, d):
                if np.unique(train_data[:, self.pi[j][i]]).size == 1:
                    self.unique_label[j][i] = train_data[0, self.pi[j][i]]
                else:
                    self.clf[j][i] = LogisticRegression(solver='liblinear',
                                                        multi_class='ovr', C=1000000000).fit(train_data[:, self.pi[j][:i]].reshape([-1, i]), train_data[:, self.pi[j][i]])
                    # self.clf[j][i] = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=10, warm_start=True, C=1000000000)
                    #
                    # best_val_LL = -np.Inf
                    # best_clf = None
                    # for iter in range(1): # config.max_iter):
                    #     self.clf[j][i].fit(train_data[:, self.pi[j][:i]].reshape([-1, i]), train_data[:, self.pi[j][i]])
                    #     LL = np.mean(self.clf[j][i].predict_log_proba(validation_data[:, self.pi[j][:i]].reshape([-1, i])))
                    #     if LL > best_val_LL:
                    #         best_val_LL = LL
                    #         best_clf = copy.deepcopy(self.clf[j][i])
                    #
                    # self.clf[j][i] = copy.deepcopy(best_clf)

        print('fit finish')
        sys.stdout.flush()

    def predict(self, test_data, verbose=False):
        d = test_data.shape[1]
        n = test_data.shape[0]
        eps = 0.00001

        log_probs = np.zeros([n, config.number_of_permutations])
        for j in range(config.number_of_permutations):
            if self.unique_label[j][0] is not None:
                log_probs[:, j] = np.ones([n]) * np.log(1-eps)
                log_probs[test_data[:,self.pi[j][0]] != self.unique_label[j][0], j] = np.log(eps)
            else:
                log_probs[:, j] = np.log(self.mu_0[j]) * np.ones([n])
                log_probs[test_data[:,self.pi[j][0]] == 0, j] = np.log(1 - self.mu_0[j])
            for i in range(1, d):
                if self.unique_label[j][i] is not None:
                    # tmp = np.zeros([n])
                    # tmp[test_data[:, self.pi[j][i]] != self.unique_label[j][i]] = -np.Inf
                    tmp = np.ones([n]) * np.log(1-eps)
                    tmp[test_data[:, self.pi[j][i]] != self.unique_label[j][i]] = np.log(eps)
                else:
                    lp = self.clf[j][i].predict_log_proba(test_data[:, self.pi[j][:i]].reshape([-1, i]))
                    tmp = np.zeros([n])
                    ind0 = (test_data[:, self.pi[j][i]] == self.clf[j][i].classes_[0])
                    ind1 = (test_data[:, self.pi[j][i]] == self.clf[j][i].classes_[1])
                    tmp[ind0] = lp[ind0, 0]
                    tmp[ind1] = lp[ind1, 1]
                log_probs[:, j] += tmp
        return logsumexp(log_probs, axis=1) - np.log(config.number_of_permutations)

    def generate(self, n):
        pass