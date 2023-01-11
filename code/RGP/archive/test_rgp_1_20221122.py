# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:15:04 2022

@author: lomp
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import tqdm as tqdm
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as PI
from recursiveGP import recursiveGP
import itertools
import gpflow
from gpflow.utilities import print_summary
import json, os
from sklearn.model_selection import train_test_split

TRAIN_DATA = "train_data.csv"
TEST_DATA  = "test_data.csv"
test_size_proportion = 0.2
batch                = 20
iterations           = 100
num_intervals        = 3
FullGP = False
SAVE = False
PLOTTING=False
# dataset 4 - x-value
dataframe = np.loadtxt("datasets/dataset4.csv",delimiter=",")


#X_values    = dataframe[:,:input_size]
#Y_values    = dataframe[:, input_size:input_size+1]
offs=0
input_size=6
yoffset =6
X_values    = dataframe[:,offs:offs+input_size]
Y_values    = dataframe[:,yoffset:yoffset+1]



#normalize input data 
ds_offset=[np.mean(X_values[:,i:i+1]) for i in range(input_size)]
ds_length=[np.std(X_values[:,i:i+1]) for i in range(input_size)]  
       
for i in range(input_size):
    if(ds_length[i]!=0):
        X_values[:,i:i+1] = (X_values[:,i:i+1]-ds_offset[i])/ds_length[i] 
    
num_dataset         = X_values.shape[0]

if(SAVE and os.path.isfile(TEST_DATA)):
    print("read train and test data from file.")
    TestData = np.loadtxt(TEST_DATA,delimiter=",")
    TrainData = np.loadtxt(TRAIN_DATA,delimiter=",")        
    XTest = TestData[:,:input_size]
    YTest = TestData[:,input_size:input_size+1]    
    XTrain = TrainData[:,:input_size]
    YTrain = TrainData[:,input_size:input_size+1]    
else:
    print("... splitting of train- and testdata.")
    XTrain, XTest , YTrain, YTest = train_test_split(X_values,
                                                     Y_values, 
                                                     test_size=test_size_proportion, 
                                                     random_state=42)
    np.savetxt(TEST_DATA,np.hstack((XTest,YTest)),delimiter=",")
    np.savetxt(TRAIN_DATA,np.hstack((XTrain,YTrain)),delimiter=",")    
    

print("Choosing random base vectors equidistant from each other")   
lb=[np.min(X_values[:,i]) for i in range(input_size)]
ub=[np.max(X_values[:,i]) for i in range(input_size)]  
list_base_coordinates=[ list(np.linspace(lb[i], ub[i], num_intervals))  for i in range(input_size)]
list_base_vectors=[]
for b in itertools.product(*list_base_coordinates):
     list_base_vectors.append(b)

X_base = np.array(list_base_vectors)
num_base_vectors =  len(list_base_vectors) # 2^6 


if(FullGP):
    print("Create full GP")    
    modelGP = gpflow.models.GPR(data=(XTrain[:num_base_vectors], YTrain[:num_base_vectors]), 
                        kernel=gpflow.kernels.SquaredExponential(variance=1.0,lengthscales=list(2*(np.array(ub)-np.array(lb))/num_intervals)), 
                        mean_function=None)
    modelGP.likelihood.variance.assign(1E-05)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(modelGP.training_loss, 
                          modelGP.trainable_variables)
    print(modelGP.kernel.lengthscales)

#LS=[12.105,12.692,20.176,783263.16,32.4,1226044.06]
#LS = list(modelGP.kernel.lengthscales.numpy())
#VAR= modelGP.kernel.variance.numpy()
#modelRGP = recursiveGP(variance = VAR, lengthscales = LS, sigma = 0.09)         
#modelRGP = recursiveGP(variance = 2.0, lengthscales = LS, sigma = 0.09)         
modelRGP = recursiveGP(variance = 1.0, lengthscales = list(2*(np.array(ub)-np.array(lb))/num_intervals), sigma = 0.0)         

print("Initiate RGP model with  "+ str(num_base_vectors) + " base vectors.")   
modelRGP.initialise(X_base)

print("RMSE on Test Data before training:")

YTest_pred_RGP, _ = modelRGP.predict(XTest)
print("rmse Loss (RGP): "+str(modelRGP.lossRMSE(YTest,YTest_pred_RGP)))


print("Start recursive GP")

for i in tqdm.trange(iterations):
    I = np.random.choice(num_dataset, batch, replace=False)  
    modelRGP.recursiveGP(X_values[I], Y_values[I]) 



############

print("RMSE on Test Data after training:")
YTest_pred_RGP, _ = modelRGP.predict(XTest)
print("rmse Loss (RGP): "+str(modelRGP.lossRMSE(YTest,YTest_pred_RGP)))
if(FullGP):
    YTest_pred_GP,_  = modelGP.predict_f(XTest)
    print("rmse Loss (GP): "+str(modelRGP.lossRMSE(YTest,YTest_pred_GP)))


def plot_histogram(ax, YY, YYp, nbins=50):
     ax.hist(abs(YY.reshape(1,-1)-YYp.reshape(1,-1)),bins=nbins)
     return(ax)
    
if(FullGP and PLOTTING):
    figure, ax = plt.subplots(1,2, figsize = (10,10))
    ax[0]=plot_histogram(ax[0],YTest,YTest_pred_RGP)
    ax[1]=plot_histogram(ax[1],YTest,YTest_pred_GP)
    plt.show()

def objective_fn(ls,var=1.0, sn=0.1,iterations=100,batch=10):
    modelRGP = recursiveGP(variance = var, lengthscales = ls, sigma = sn)        
    modelRGP.initialise(X_base)
    for i in range(iterations):
        I = np.random.choice(num_dataset, batch, replace=False)  
        modelRGP.recursiveGP(X_values[I], Y_values[I]) 
    YTest_pred_RGP, _ = modelRGP.predict(XTest)
    return(modelRGP.lossRMSE(YTest,YTest_pred_RGP)[0])





# def plot_prediction(ax,XX,YY, YYp, C, labelstr):
#     N=len(XX)
#     ax.plot(range(N), YY , "b--")
#     ax.plot(range(N), YYp, "r--")   
#     ax.set_xlabel(labelstr) 
#     D=np.diag(C)
#     if(np.min(D)<0):
#         print(np.min(D))
#     C=np.sqrt(D)
#     C.reshape(-1,1)
#     CI = 1.96 * C/np.sqrt(N)  
#     CI = CI.reshape(-1,1)
#     Ylower = (YYp-CI).reshape((N,))
#     Yupper = (YYp+CI).reshape((N,))
#     ax.fill_between(range(N), Ylower, Yupper , color='orange', alpha=.25)   
#     return(ax)
        

# def plot_predict_diagram(model,XX,YY,ax1, ax2,label):
#     YYp, C = model.predict(XX)    
#     YYp = YYp.numpy()        
#     ax1 = plot_histogram(ax1,YY,YYp)
#     ax2 = plot_prediction(ax2,XX, YY, YYp, C, labelstr=label)
    
