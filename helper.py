"""
A series of helper functions used throughout the creating models from scratch.
"""
import numpy as np


class BaseRegression:
    '''
    Base for all regression models
    '''

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):

            # get predicted
            y_predicted = self._approximation(X, self.weights, self.bias)

            # derivative with respect to w
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))

            # derivative with respect to b
            db = (1/n_samples) * np.sum(y_predicted - y)

            # update weights and bias
            self._update(dw, db)

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _update(self, dw, db):
        # update rules: new weight
        self.weights -= self.lr * dw

        # update rules: new bias
        self.bias -= self.lr * db

    def _approximation(self, X, w, b):
        raise NotImplementedError()

    def _predict(self, X, w, b):
        raise NotImplementedError()


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
