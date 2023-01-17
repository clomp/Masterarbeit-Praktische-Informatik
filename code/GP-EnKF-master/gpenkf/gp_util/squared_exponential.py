import numpy as np
from scipy.spatial.distance import cdist

from gpenkf.gp_util.covariance_function import CovarianceFunction

#sqeuclidean to seuclidean   


class SquaredExponential(CovarianceFunction):

    def __init__(self, variance, lengthscale):
        self.variance = variance
        self.lengthscale = lengthscale

    @classmethod
    def from_parameters_vector(cls, vec):
        variance, lengthscale = vec[0], vec[1:]
        cov_func = cls(variance=variance, lengthscale=lengthscale)
        return cov_func

    def eval_cov_matrix(self, x1, x2):
        return self.variance * np.exp(-cdist(x1, x2, 'seuclidean', V=2 * self.lengthscale ** 2))

    def predict(self, x_new, x_g, noise_variance, g):
        # TODO estimate lengthscale for each dimension
        K_new_g = self.variance * np.exp(-cdist(x_new, x_g, 'seuclidean', V=2 * self.lengthscale ** 2))

        K_g_g = self.variance * np.exp(-cdist(x_g, x_g, 'seuclidean', V=2 * self.lengthscale ** 2))

        K_new_new = self.variance * np.exp(-cdist(x_new, x_new, 'seuclidean', V=2 * self.lengthscale ** 2))

        pred_mean = np.dot(K_new_g, np.linalg.solve(K_g_g + noise_variance * np.eye(x_g.shape[0]), g))
        pred_cov = K_new_new - np.dot(K_new_g,
                                      np.linalg.solve(K_g_g + noise_variance * np.eye(x_g.shape[0]), K_new_g.T))
        return pred_mean, pred_cov

    def log_likelihood(self, y, x_g, noise_variance):
        if(type(self.lengthscale) != np.ndarray):
            a = np.array([self.lengthscale])
        else:
            a=self.lengthscale
        
        K_g_g = self.variance * np.exp(-cdist(x_g, x_g, 'seuclidean', V=2 * a ** 2))

        return -0.5 * (np.dot(y.T, np.linalg.solve(K_g_g + noise_variance * np.eye(x_g.shape[0]), y))
                       + np.log(np.linalg.det(K_g_g + noise_variance * np.eye(x_g.shape[0])))
                       + x_g.shape[0] * np.log(2 * np.pi))