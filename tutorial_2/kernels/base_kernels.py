import numpy as np


class Kernel:
    """
    Abstract kernel class
    """
    def __init__(self):
        pass

    def kernel(self ,X ,Y):
        pass


class Linear(Kernel):
    def __init__(self):
        super().__init__()

    def kernel(self, X, Y):
        # Input vectors X and Y of shape Nxd and Mxd
        return X.dot(Y.T)  ## Matrix of shape NxM


class RBF(Kernel):
    def __init__(self, sigma=1.):
        super().__init__()
        self.sigma = sigma  # the variance of the kernel

    def kernel(self, X, Y):
        # Input vectors X and Y of shape Nxd and Mxd

        norm_matrix = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2) ** 2
        return np.exp(- norm_matrix / (2 * self.sigma ** 2))  # Matrix of shape NxM
