import numpy as np
from scipy.linalg import cho_factor, cho_solve


class KernelRR:
    def __init__(self, kernel, lmbda, verbose=False):
        self.lmbda = lmbda
        self.kernel = kernel
        self.alpha = None
        self.b = None
        self.support = None
        self.type = 'ridge'
        self.verbose = verbose

    def fit(self, X, y):
        if self.verbose:
            print("Start training")

        N = X.shape[0]
        self.support = X
        self.b = np.mean(y)

        self.alpha = cho_solve(cho_factor(self.kernel(X, X) + self.lmbda * N * np.eye(N)), y - self.b)
        if self.verbose:
            print("End training")

    def regression_function(self, x):
        return self.kernel(x, self.support) @ self.alpha

    def predict(self, X):
        return self.regression_function(X) + self.b