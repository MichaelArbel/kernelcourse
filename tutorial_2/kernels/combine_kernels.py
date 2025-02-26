from kernels.base_kernels import Kernel
import numpy as np


class SumKernel(Kernel):
    """
    The weighted sum kernel of an arbitrary number of subkernels.
    """
    def __init__(self, kernels, alphas=None):
        super().__init__()
        self.kernels = kernels
        self.N = len(self.kernels)
        if alphas is None:
            self.alphas = np.ones(self.N)
        else:
            self.alphas = alphas

    def kernel(self, X, Y):
        return sum([alpha * kernel(X, Y) for alpha, kernel in zip(self.alphas, self.kernels)])
