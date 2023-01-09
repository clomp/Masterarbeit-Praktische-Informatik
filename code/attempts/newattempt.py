# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:14:00 2022

@author: lomp
"""


from IPython import get_ipython
get_ipython().magic('reset -sf')

import tqdm as tqdm
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as PI
import itertools
import math
from recursiveGP import recursiveGP
import rgp_utilities 
import gpflow
from gpflow.utilities import print_summary
import json, os
from sklearn.model_selection import train_test_split

TEST_DATA = "Test_Data_"
TRAIN_DATA = "Train_Data_"
META_DATA = "Meta_Data_"

 
def BuildFullGP(dataset, input_size, output_size=3, scalar=1000, xlims=1.0):
    print("#### "+dataset+" ####")
    dataframe = np.loadtxt("datasets/"+dataset+".csv",delimiter=",")
    num_data = dataframe.shape[1]
    
    ds_offset=[np.mean(dataframe[:,i:i+1]) for i in range(num_data)]
    ds_length=[np.std(dataframe[:,i:i+1]) for i in range(num_data)]  
    
    for i in range(num_data):
        if(ds_length[i]!=0):
            dataframe[:,i:i+1] = (dataframe[:,i:i+1]-ds_offset[i])/ds_length[i]        
    
    X_values    = dataframe[ : , 0 : input_size]
    Y_values    = dataframe[ : , input_size : ]
    
    XTrain, XTest , YTrain, YTest = train_test_split(X_values,Y_values, test_size=0.2, random_state=42)
    
    np.savetxt(TEST_DATA+dataset+".csv",np.hstack((XTest,YTest)),delimiter=",")
    np.savetxt(TRAIN_DATA+dataset+".csv",np.hstack((XTrain,YTrain)),delimiter=",") 
    out_file = open(META_DATA+dataset+".json", "w")
    json.dump((ds_offset,ds_length), out_file)
    
    modelsGP = []
    YTest_prediction=[]
    LS=[0.5]*input_size
    for i in range(output_size):
        print(" = "*20)
        print("build and train "+str(i)+"th full GP with "+str(XTrain.shape[0])+" training data.")
        YTrn = YTrain[:,i].reshape((XTrain.shape[0],1))
        model_aux=gpflow.models.GPR(data=(XTrain, YTrn), 
                    kernel=gpflow.kernels.SquaredExponential(variance=1.0,lengthscales=LS), mean_function=None)
        model_aux.likelihood.variance.assign(1E-05)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model_aux.training_loss, model_aux.trainable_variables, options=dict(maxiter=50))        
        modelsGP.append(model_aux)    
        YTst_aux, _ = model_aux.predict_f(XTest)
        YTest_prediction.append(YTst_aux)
        
    diffs=[]
    for i in range(output_size):
        YTst = YTest[:,i].reshape((XTest.shape[0],))
        if(ds_length[i]!=0):
            YTst = YTst*ds_length[i+input_size] + ds_offset[i+input_size]
            YTest_prediction[i] = YTest_prediction[i] * ds_length[i+input_size]+ds_offset[i+input_size]
        diff = scalar*np.abs(YTst - YTest_prediction[i].reshape((XTest.shape[0],)))
        diffs.append(diff)
    rgp_utilities.plotting(np.array(diffs),dataset,rgp_utilities.analyzeError(np.array(diffs).T,False),xlims=xlims)
        

BuildFullGP("dataset6", input_size=9) 
BuildFullGP("dataset5", input_size=6) 
BuildFullGP("dataset4", input_size=6) 


#dataframe = np.loadtxt("datasets/dataset6.csv",delimiter=","); input_size=9; output_size=6
#dataframe = np.loadtxt("datasets/dataset4.csv",delimiter=","); input_size=6;  output_size=3;  
#dataframe = np.loadtxt("datasets/dataset5.csv",delimiter=","); input_size=6;  output_size=3;  

    