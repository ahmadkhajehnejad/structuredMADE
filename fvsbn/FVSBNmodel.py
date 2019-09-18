import numpy as np
from sklearn.linear_model import LogisticRegression
import sys


class FVSBN:

    def __init__(self):
        self.mu_0 = None
        self.clf = None
        self.pi = None
        self.unique_label = None

    def fit(self, train_data, validation_data):
        print('fit start')
        d = train_data.shape[1]

        self.pi = np.random.permutation(d)
        self.unique_label = [None] * d
        if np.unique(train_data[:,self.pi[0]]).size == 1:
            self.unique_label[0] = train_data[0,self.pi[0]]
        else:
            self.mu_0 = np.sum(train_data[:, self.pi[0]] == 1) / train_data.shape[0]
        self.clf = [None] * d
        for i in range(1, d):
            if np.unique(train_data[:, self.pi[i]]).size == 1:
                self.unique_label[i] = train_data[0, self.pi[i]]
            else:
                self.clf[i] = LogisticRegression(random_state=0, solver='liblinear',
                                                 multi_class='ovr').fit(train_data[:, self.pi[:i]].reshape([-1, i]),
                                                                        train_data[:, self.pi[i]])

        print('fit finish')
        sys.stdout.flush()

    def predict(self, test_data, verbose=False):
        d = test_data.shape[1]
        n = test_data.shape[0]
        eps = 0.00001
        if self.unique_label[0] is not None:
            log_probs = np.ones([n]) * np.log(1-eps)
            log_probs[test_data[:,self.pi[0]] != self.unique_label[0]] = np.log(eps)
        else:
            log_probs = np.log(self.mu_0) * np.ones([n])
            log_probs[test_data[:,self.pi[0]] == 0] = np.log(1 - self.mu_0)
        for i in range(1, d):
            if self.unique_label[i] is not None:
                # tmp = np.zeros([n])
                # tmp[test_data[:, self.pi[i]] != self.unique_label[i]] = -np.Inf
                tmp = np.ones([n]) * np.log(1-eps)
                tmp[test_data[:, self.pi[i]] != self.unique_label[i]] = np.log(eps)
            else:
                lp = self.clf[i].predict_log_proba(test_data[:, self.pi[:i]].reshape([-1, i]))
                tmp = np.zeros([n])
                ind0 = (test_data[:, self.pi[i]] == self.clf[i].classes_[0])
                ind1 = (test_data[:, self.pi[i]] == self.clf[i].classes_[1])
                tmp[ind0] = lp[ind0, 0]
                tmp[ind1] = lp[ind1, 1]
            log_probs += tmp
        return log_probs

    def generate(self, n):
        pass