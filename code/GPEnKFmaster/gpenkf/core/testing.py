from GPEnKFmaster.gpenkf.experiments.experiment_runner import ExperimentRunner
from GPEnKFmaster.gpenkf.core.parameters import Parameters
from GPEnKFmaster.gpenkf.experiments.synthetic.data_generator import myDataGenerator
from RGP.dataset import dataset
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    dataset_nr    = 4
    
    grid_size     = 50
    ensemble_size = 100    
    iterations    = 200
    batch_size    = 10
    output_coord  = 0
    
    data = dataset(dataset_nr)
    data.load_dataframe(reuse=True)    
    data_provider = myDataGenerator(data, batch_size, output_coord)
    
    x =  data.CreateBasisVectors(grid_size, "JB")
    
    parameters = Parameters(T                         = iterations, 
                            sample_size               = batch_size, 
                            grid_size                 = grid_size, 
                            inducing_points_locations = x, 
                            ensemble_size             = ensemble_size, 
                            sigma_eta                 = 0.01,
                            sigma_y                   = 0.01, 
                            init_cov                  = 0.001, 
                            initial_log_gp_params     = [0, 0], 
                            initial_log_sigma         = 0,
                            log_sigma_unlearnt        = 0, 
                            gp_hyperparams_dimensionality = 2)
    
    # 'learn_enkf_gp','learn_enkf_liuwest_gp' 
    runner = ExperimentRunner(data_provider=data_provider, parameters=parameters, algorithms=['learn_enkf_gp']) 
    
    runner.run()
    
    print("\n MSE of prediction of coordinate "+str(output_coord)+" :"+str(np.mean(runner.results['gpenkf_learn_gp'].mse_history)))
    plt.plot(np.linspace(1,parameters.T, parameters.T),runner.results['gpenkf_learn_gp'].prediction_history);    
    
#    p=runner.results['gpenkf_learn_gp'].prediction_history
 #   mean=np.array([data_provider.get_error_mean(p[i]) for i in range(200)])
  #  plt.plot(np.linspace(1,parameters.T, parameters.T),mean);    
    plt.show()
    
    