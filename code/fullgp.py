# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:14:00 2022

@author: lomp
"""

import json
import gpflow
from dataset import dataset

def fullgp(dataset_nr, num_coordinates, save_hyper=False, savePDF=False, SCALING=1000):
    classname = gpflow.__name__
    dataobj = dataset(dataset_nr,classname)
    dataobj.load_dataframe()
    N=dataobj.XTrain.shape[0]
    input_size=dataobj.input_dim
    limitation = dataobj.limitations
    Lengthscales=[]
    Variances=[]
    models = []
    
    for i in range(num_coordinates):
        print(" = "*20)
        print("build and train "+str(i)+"th "+classname+" model on "+str(N)+" training data.")
        if(limitation is None):        
            m  = gpflow.models.GPR(data=(dataobj.XTrain, dataobj.YTrain[:,i:i+1]),
                                          kernel=gpflow.kernels.SquaredExponential(variance=1.0,lengthscales=[0.5]*input_size), 
                                          mean_function=None)                
        else:        
            m  = gpflow.models.GPR(data=(dataobj.XTrain[:limitation], dataobj.YTrain[:limitation,i:i+1]), 
                                        kernel=gpflow.kernels.SquaredExponential(variance=1.0,lengthscales=[0.5]*input_size), 
                                        mean_function=None)                       
        m.likelihood.variance.assign(1E-05)
        opt    = gpflow.optimizers.Scipy()
        _      = opt.minimize(m.training_loss, m.trainable_variables)        
        Lengthscales.append(list(m.kernel.lengthscales.numpy()))
    
        varianz = m.kernel.variance.variables[0].value().numpy()
        if(varianz < 0):
            print("negative variance"+str(varianz))
            Variances.append(1.0)
        else:
            Variances.append(varianz)
        models.append(m)    

    if(save_hyper):
        print("Saving hyperparameter")
        out_file = open(dataobj.meta_filename, "w")
        json.dump([dataobj.ds_offset,dataobj.ds_length,Lengthscales,Variances], out_file)
        out_file.close()   

    print("Calculate Predictions and analyze error")   
    predictions = [m.predict_f(dataobj.XTest)[0].numpy() for m in models]
    difference, totalMSE, componentwiseErrors = dataobj.analyze_error(predictions,SCALING)
    
    print("Print error analysis")   
    dataobj.print_analysis(difference.T, totalMSE, componentwiseErrors,savePDF=savePDF)  
    print("---- END ----")
