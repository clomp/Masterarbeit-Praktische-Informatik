from gpenkf.experiments.experiment_runner import ExperimentRunner, Results
from gpenkf.core.parameters import Parameters
from gpenkf.experiments.synthetic.mydata_generator import myDataGenerator
from gpenkf.core import DualGPEnKF
from dataset import dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import argparse

def testGPEnKF(dataset_nr, num_coordinates=3, grid_size=50, ensemble_size=100, iterations=200, batch_size=10, savePDF=False, SCALING=1000):    
    
    rgpmodel=DualGPEnKF
    
    classname = rgpmodel.__name__
    print("---- TESTSUITE for model "+classname+ " ----")
    
    print("Load dataset ", dataset_nr)
    data = dataset(dataset_nr, classname)
    data.load_dataframe(load=False)    
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
    
    print("Create ", num_coordinates, " models of class ", classname)
    models=[]
    results=[]
    data_provider=[]
    for i in range(num_coordinates):
        data_provider.append(myDataGenerator(data, batch_size, i))
        models.append(rgpmodel(parameters, learn_gp_parameters=True, learn_sigma=True))
        results.append(Results(T=parameters.T, params_dimensionality=parameters.hyperparams_dimensionality + 1,
                                                           grid_size=parameters.grid_size,
                                                           ensemble_size=parameters.ensemble_size))
        
    print("Training of ", num_coordinates, " models of class ", classname)
    
    for t in tqdm(range(parameters.T)):         
             for i in range(num_coordinates):
                 x_new, f_new_noisy = data_provider[i].generate_sample()
                 start_time = time.time()
                 models[i].run_iteration(x_new, f_new_noisy)
                 results[i].time[t] = time.time() - start_time
                     
                 results[i].g_mean_history[t] = models[i].get_g_mean().T
    
                 results[i].likelihood_history[t] = models[i].compute_log_likelihood(data_provider[i].x_validation, data_provider[i].f_validation)
                              
                 results[i].mse_history[t] = models[i].compute_mse(data_provider[i].x_validation, data_provider[i].f_validation)
                                 
                 p = models[i].compute_prediction(data_provider[i].x_validation)
                 results[i].prediction_history[t] = SCALING*data_provider[i].get_error_mean(p.reshape(-1,1))    
    
    print("Calculate Predictions and analyze error")   
    predictions=[]
    for i in range(num_coordinates):
        predictions.append(models[i].compute_prediction(data_provider[i].x_validation).reshape(-1,1))
        
    difference, totalMSE, componentwiseErrors = data.analyze_error(predictions,SCALING)
    
    print("Print error analysis")   
    data.print_analysis(difference.T, totalMSE, componentwiseErrors, savePDF=savePDF)  
    
    for i in range(num_coordinates):
        plt.plot(np.linspace(1,parameters.T, parameters.T),results[i].prediction_history);
    plt.show()
    return(totalMSE)


def parameter_search(dataset_nr, grid_range, ensemble_range):
    results = np.zeros((len(grid_range),len(ensemble_range)))
    grid_low = grid_range.start
    grid_step = grid_range.step
    ensemble_low = ensemble_range.start
    ensemble_step = ensemble_range.step
    print(grid_low, grid_step, ensemble_low,ensemble_step)
    for grid in grid_range:
        for ensemble in ensemble_range:            
            i=int((grid-grid_low)/grid_step)
            j=int((ensemble-ensemble_low)/ensemble_step)
            print(grid,ensemble,i,j)
            results[i,j] = testGPEnKF(dataset_nr,grid_size=grid, ensemble_size=ensemble)
            
    return(results)
    
#---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='testGPEnKF',
                                      description='Runs a test on datasets for the GPEnKF model.',
                                      epilog="implemented by Christian Lomp")

    parser.add_argument('-d', '--dataset', help='indicates the dataset number', default="7")
                            
    args = parser.parse_args()

    dataset_nr = int(args.dataset)
    
    testGPEnKF(dataset_nr, num_coordinates=3, grid_size=50, ensemble_size=100, iterations=200, batch_size=10, savePDF=False, SCALING=1000)
    
    
    