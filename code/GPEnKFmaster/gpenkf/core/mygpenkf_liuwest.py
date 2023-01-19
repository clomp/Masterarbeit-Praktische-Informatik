"""
Dual GP-EnKF model
"""

import numpy as np
import time
from tqdm import tqdm
from gpenkf.gp_util.squared_exponential import SquaredExponential
from gpenkf.experiments.data_provider import DataProvider
from gpenkf.experiments.synthetic.data_generator import DataGenerator
from gpenkf.core.parameters import Parameters

class DualGPEnKF:
    """
    Dual GP-EnKF model. Has separate ensembles for parameters and state.

    :param parameters: The :class:`~gpenkf.core.parameters.parameters` parameters
    :param learn_gp_parameters: indicator if the GP hyperparameters should be learnt
    :param learn_sigma: indicator if the noise variance should be learnt
    """

    def __init__(self, parameters):
        
        self.parameters                 = parameters
        self.params_dimensionality      = parameters.hyperparams_dimensionality+1                
        self.ensemble_size              = parameters.ensemble_size
        self.sample_size                = parameters.sample_size
        self.sigma_y                    = parameters.sigma_y
        self.sigma_eta                  = parameters.sigma_eta * np.eye(self.params_dimensionality, dtype=np.double)
        self.inducing_points_locations  = parameters.inducing_points_locations
        self.initial_log_gp_params      = parameters.initial_log_gp_params
        self.initial_log_sigma          = parameters.initial_log_sigma
        self.min_exp                    = -20

        params_mean = np.array([self.initial_log_gp_params[0], self.initial_log_gp_params[1], self.initial_log_sigma])
        params_cov  = self.parameters.init_cov * np.eye(self.params_dimensionality)
        
        self.params_ensemble = np.random.multivariate_normal(mean=params_mean,
                                                                 cov=params_cov,
                                                                 size=self.ensemble_size)
        
        self.grid_size                  = parameters.grid_size
        g_mean                          = np.zeros((self.grid_size,))
        g_cov                           = np.eye(self.grid_size)        
        self.g_ensemble = np.random.multivariate_normal(mean=g_mean, cov=g_cov, size=self.ensemble_size)

        self.predictions = np.zeros((self.ensemble_size, self.sample_size))

        self.y_trajectories = np.zeros((self.ensemble_size, self.sample_size))

    def run_iteration(self, x_new, f_new_noisy):
        """
        Perform one iteration of predict-update loop.

        :param x_new: locations of new observations
        :param f_new_noisy: values of new observations
        """

        self.predict(x_new, f_new_noisy)        
        self.update_state(x_new)

    def predict(self, x_new, f_new_noisy):
        """
        Predict step of the predict-update loop.

        :param x_new: locations of new observations
        :param f_new_noisy: values of new observations
        """
        self.__sample_parameters()
        self.__predict_observations(x_new)
        self.__compute_noisy_trajectories(f_new_noisy)


    def update_parameters(self):
        """
        Update parameters step of the predict-update loop.
        """
        sigma_eta_y     = self.__compute_cross_covariance_of_parameters_and_predictions()
        sigma_y_y       = self.__compute_forecast_error_covariance_matrix_of_predictions()
        kalman_gain_eta = self.__compute_kalman_gain_for_parameters(sigma_eta_y, sigma_y_y)
        self.__update_parameters_internal(kalman_gain_eta)

    def update_state(self, x_new):
        """
        Update state step of the predict-update loop.
        """
        self.__predict_observations(x_new)
        sigma_g_y       = self.__compute_cross_covariance_of_state_and_prediction()
        sigma_y_y       = self.__compute_forecast_error_covariance_matrix_of_predictions()
        kalman_gain_g   = self.__compute_kalman_gain_for_state(sigma_g_y, sigma_y_y)
        self.__update_state_internal(kalman_gain_g)

    def __sample_parameters(self):        
        noise = np.random.multivariate_normal( 
                                mean=np.zeros(shape=(self.params_dimensionality,)), 
                                cov=self.sigma_eta, 
                                size=self.ensemble_size)
        
        self.params_ensemble = self.params_ensemble + noise        
        
    def __predict_observations(self, x_new):
        for ens_idx in range(self.ensemble_size):
            mean = self.__predict_at_obs(x_new, self.params_ensemble[ens_idx], self.g_ensemble[ens_idx])
            self.predictions[ens_idx] = mean
            
    def __predict_at_obs(self, x_sample, log_params, g_mean):
        log_gp_params = log_params[:-1]
        log_sigma = log_params[-1]

        cov_func = SquaredExponential.from_parameters_vector(np.exp(log_gp_params))
        mean, _ = cov_func.predict(x_sample, self.inducing_points_locations, np.exp(log_sigma), g_mean)
        return mean
    
    def get_prediction(self, x_sample):
        log_params  = np.mean(self.params_ensemble, axis=0)
        g_mean      = np.mean(self.g_ensemble, axis=0)
        return(self.__predict_at_obs(x_sample, log_params, g_mean))

    def __compute_noisy_trajectories(self, f_new_noisy):
        for ens_idx in range(self.ensemble_size):
            noise = np.random.normal(loc=0., scale=self.sigma_y, size=(self.sample_size,))
            self.y_trajectories[ens_idx] = (f_new_noisy + noise).T

    def __compute_cross_covariance_of_parameters_and_predictions(self):
        predictions_mean        = np.mean(self.predictions, axis=0)
        params_ensemble_mean    = np.mean(self.params_ensemble, axis=0)
        sigma_eta_y             = np.zeros((self.params_dimensionality, self.sample_size))
        for ens_idx in range(self.ensemble_size):
            sigma_eta_y += np.outer(self.params_ensemble[ens_idx] - params_ensemble_mean, self.predictions[ens_idx] - predictions_mean)
        sigma_eta_y /= (self.ensemble_size - 1)

        return sigma_eta_y

    def __compute_forecast_error_covariance_matrix_of_predictions(self):
        predictions_mean        = np.mean(self.predictions, axis=0)
        sigma_y_y               = np.zeros((self.sample_size, self.sample_size))
        for ens_idx in range(self.ensemble_size):
            sigma_y_y += np.outer(self.predictions[ens_idx] - predictions_mean, self.predictions[ens_idx] - predictions_mean)
        sigma_y_y /= (self.ensemble_size - 1)

        return sigma_y_y

    def __compute_kalman_gain_for_parameters(self, sigma_eta_y, sigma_y_y):
        kalman_gain_eta = np.matmul(sigma_eta_y, np.linalg.inv(sigma_y_y + self.sigma_y * np.eye(self.sample_size)))

        return kalman_gain_eta

    def __update_parameters_internal(self, kalman_gain_eta):
        for ens_idx in range(self.ensemble_size):
            self.params_ensemble[ens_idx] += np.matmul(kalman_gain_eta, self.y_trajectories[ens_idx] - self.predictions[ens_idx])
        self.__check_params_ensemble()

    def __compute_cross_covariance_of_state_and_prediction(self):
        predictions_mean = np.mean(self.predictions, axis=0)
        g_ensemble_mean = np.mean(self.g_ensemble, axis=0)
        sigma_g_y = np.zeros((self.grid_size, self.sample_size))
        for ens_idx in range(self.ensemble_size):
            sigma_g_y += np.outer(self.g_ensemble[ens_idx] - g_ensemble_mean, self.predictions[ens_idx] - predictions_mean)
        sigma_g_y /= (self.ensemble_size - 1)

        return sigma_g_y

    def __compute_kalman_gain_for_state(self, sigma_g_y, sigma_y_y):
        kalman_gain_g = np.matmul(sigma_g_y, np.linalg.inv(sigma_y_y + self.sigma_y * np.eye(self.sample_size)))

        return kalman_gain_g

    def __update_state_internal(self, kalman_gain_g):
        for ens_idx in range(self.ensemble_size):
            self.g_ensemble[ens_idx] += np.matmul(kalman_gain_g, self.y_trajectories[ens_idx] - self.predictions[ens_idx])
        self.__check_g_ensemble()

    def __check_params_ensemble(self):
        if not np.all(np.isfinite(self.params_ensemble)):
            raise ValueError('params are not finite')
        self.params_ensemble = np.clip(self.params_ensemble, self.min_exp, None)

    def __check_g_ensemble(self):
        if not np.all(np.isfinite(self.g_ensemble)):
            raise ValueError('g are not finite')

    def get_eta_ensemble(self):
        """
        :return: Ensemble of logarithms of GP log hyperparameters
        """
        return self.params_ensemble[:, -1]


    def get_log_mean_params(self):
        """
        :return: logarithm of mean GP hyperparameters, logarithm of mean noise variance
        """
        params          = self.params_ensemble.mean(axis=0)
        log_gp_params   = params[:-1]
        log_sigma       = params[-1]

        return log_gp_params, log_sigma

    def get_g_mean(self):
        """
        :return: mean of state ensemble
        """
        return np.mean(self.g_ensemble, axis=0)

    def compute_nmse(self, x_sample, f_true_sample):
        """
        :param x_sample: location of the points to predict
        :param f_true_sample: true value at sample points
        :return: NMSE between predicted and true values at sample points
        """
        eta_mean    = np.mean(self.params_ensemble, axis=0)
        g_mean      = np.mean(self.g_ensemble, axis=0)

        mean        = self.__predict_at_obs(x_sample, eta_mean, g_mean)

        return np.mean(np.sqrt((mean-f_true_sample)**2)/np.sqrt(f_true_sample**2))
    
    def predict_f(self, x_sample):
        """
        :param x_sample: location of the points to predict
        :param f_true_sample: true value at sample points
        :return: NMSE between predicted and true values at sample points
        """
        eta_mean    = np.mean(self.params_ensemble, axis=0)
        g_mean      = np.mean(self.g_ensemble, axis=0)

        mean        = self.__predict_at_obs(x_sample, eta_mean, g_mean)

        return (mean)    

    def compute_log_likelihood(self, x_sample, f_true_sample):
        """
        :param x_sample: location of the points to predict
        :param f_true_sample: true value at sample points
        :return: log likelihood of true values at sample points given the estimated model
        """
        log_gp_params, log_sigma    = self.get_log_mean_params()
        cov_func                    = SquaredExponential.from_parameters_vector(np.exp(log_gp_params))
        return cov_func.log_likelihood(f_true_sample, x_sample, np.exp(log_sigma))

class Results(object):
    def __init__(self, T, params_dimensionality, grid_size, ensemble_size):
        self.eta_mean_history = np.zeros((T, params_dimensionality-1))
        self.sigma_mean_history = np.zeros((T))
        self.g_mean_history = np.zeros((T, grid_size))
        self.likelihood_history = np.zeros((T,))
        self.nmse_history = np.zeros((T,))
        self.time = np.zeros((T,))
        self.eta_last_ensemble = np.zeros((ensemble_size, params_dimensionality))


def run(parameters, data_provider,model,result):
    for t in tqdm(range(parameters.T)):
        x_new, f_new_noisy = data_provider.generate_sample()
    
        start_time = time.time()
        
        model.run_iteration(x_new, f_new_noisy)
        
        result.time[t] = time.time() - start_time
        result.eta_mean_history[t], result.sigma_mean_history[t] = model.get_log_mean_params()
        result.g_mean_history[t] = model.get_g_mean().T
        result.likelihood_history[t] = model.compute_log_likelihood(data_provider.x_validation, data_provider.f_validation)        
        result.nmse_history[t] = model.compute_nmse(data_provider.x_validation, data_provider.f_validation)    
        result.eta_last_ensemble = model.get_eta_ensemble()
    
    return(result)

sample_size = 5
data_provider = DataGenerator(borders=[-10, 10],
                               sample_size=sample_size,
                               f=lambda x: x / 2 + (25 * x) / (1 + x ** 2) * np.cos(x),
                               noise=0.01,
                               validation_size=20)

grid_size = 51
x = np.linspace(-10, 10, grid_size)
x = np.expand_dims(x, axis=1)

parameters = Parameters(T=200, 
                         sample_size=sample_size, 
                         grid_size=grid_size, 
                         inducing_points_locations=x, 
                         ensemble_size=100, 
                         sigma_eta=0.1,
                         sigma_y=0.1, 
                         init_cov=0.01, 
                         initial_log_gp_params=[0, 0], 
                         initial_log_sigma=0,
                         log_sigma_unlearnt=0, 
                         gp_hyperparams_dimensionality=2)

result = Results(T=parameters.T, params_dimensionality=parameters.hyperparams_dimensionality + 1, grid_size=parameters.grid_size, ensemble_size=parameters.ensemble_size)

model = DualGPEnKF(parameters=parameters)

run(parameters, data_provider, model, result)
import matplotlib.pyplot as plt
plt.plot(np.linspace(1,parameters.T, parameters.T),result.nmse_history)
plt.show()

#xg=np.linspace(-10,10,parameters.T)
xg=x
plt.plot(xg,data_provider.f(xg)); 
plt.plot(xg,model.predict_f(np.array(xg).reshape(-1,1)),color="red"); plt.show()
