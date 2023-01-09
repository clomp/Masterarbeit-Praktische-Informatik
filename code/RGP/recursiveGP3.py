# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:33:55 2022

@author: lomp
"""

#from IPython import get_ipython
#get_ipython().magic('reset -sf')

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
from sklearn.cluster import KMeans

TEST_DATA = "Test_Data_"
TRAIN_DATA = "Train_Data_"
META_DATA = "Meta_Data_"

def trainRGP(models, j, XTrain, YTrain, batch, num_models,random=True):
    if(random):
        I = np.random.choice(XTrain.shape[0], batch, replace=False)  
    else:
        num_train=XTrain.shape[0]
        a= (j*batch)%num_train
        b=((j+1)*batch)%num_train
        if(a<b):
            I = np.array(range(a,b))
        else:
            I = np.hstack((range(a,num_train),range(0,b)))        
    for i in range(num_models): 
        models[i].recursiveGP(XTrain[I], YTrain[I,i:i+1]) 

def getPrediction(models, XTest):
    Prediction=[]
    for model in models:
        Ypred, _ = model.predict(XTest)
        Prediction.append(Ypred.numpy())
    return(Prediction)

def normalize(A,length,offset):
    return((A-offset)/length)

def denormalize(A,length,offset):
    return((A*length)+offset)



# def CreateBasisVectors(XTrain, input_size, lb, ub, num_base_vectors=20):        
#     list_base_coordinates=[ np.linspace(lb[i], ub[i], num_base_vectors)  for i in range(input_size)]
#     for ls in list_base_coordinates:
#         np.random.shuffle(ls)
#     return(np.vstack(tuple(list_base_coordinates)).T)


RANDOMTRAINING=True

#dataset="dataset4" ; input_size=6; limitation=None;
#dataset="dataset5" ; input_size=6; limitation=None;
#dataset="dataset6" ; input_size=9; limitation=None;
dataset="dataset7" ; input_size=9; limitation=None;
#dataset="dataset1" ; input_size=6; limitation=2000;
#dataset="dataset2" ; input_size=12; limitation=2000;
#dataset="dataset3" ; input_size=12; limitation=2000;

LOADING=True
output_size=3
num_base_vectors=30
iterations=100
batch=10
scalar=1000
xlims=1.0


print("#### "+dataset+" ####")

if(LOADING): 
    TestData  = np.loadtxt(TEST_DATA+dataset+".csv",delimiter=",")
    XTest  = TestData[ : , : input_size]
    YTest  = TestData[ : , input_size: ]
    TrainData = np.loadtxt(TRAIN_DATA+dataset+".csv",delimiter=",")
    XTrain  = TrainData[ : , : input_size]
    YTrain  = TrainData[ : , input_size: ]
    f=open(META_DATA+dataset+".json")
    ds_offset, ds_length, Lengthscales, Variances = json.load(f)
    f.close()
else:
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
    Lengthscales=[0.5]*input_size
    Variances=[1.0]*input_size

lb=[np.min(XTrain[:,i]) for i in range(input_size)]
ub=[np.max(XTrain[:,i]) for i in range(input_size)] 

print("Create recursive GPs")
modelsRGP = []
for i in range(output_size):
    model_aux = recursiveGP(variance = Variances[i], lengthscales = Lengthscales[i], sigma=1E-05)      
    X_base = rgp_utilities.CreateBasisVectors(XTrain, input_size, lb, ub, version="JB", num_base_vectors=num_base_vectors)
    k=0
    while(k<100 and np.linalg.cond(model_aux.covfunc(X_base,X_base))>1E+10):
        X_base = rgp_utilities.CreateBasisVectors(XTrain, input_size, lb, ub, version="JB", num_base_vectors=num_base_vectors)
        k=k+1
    if(k>=100):
        print("Coudlnt get good base vectors for model "+str(i))
    model_aux.initialise(X_base)    
    modelsRGP.append(model_aux)
    
print("Recursive training ...")    

for j in tqdm.trange(iterations):           #for j in range(iterations):               
    trainRGP(modelsRGP, j, XTrain, YTrain, batch, output_size,random=RANDOMTRAINING)


diffs = []
NTest = YTest.shape[0]
for i in range(output_size):
    length      = ds_length[i+input_size]
    offset      = ds_offset[i+input_size]    
    YTst        = denormalize(YTest[:,i:i+1], length, offset).reshape((NTest,))    
    YTst_pred   = denormalize(modelsRGP[i].predict(XTest)[0].numpy(), length, offset).reshape((NTest,))
    diffs.append(scalar*np.abs(YTst - YTst_pred))

rgp_utilities.plotting(np.array(diffs),"fullGP "+dataset+" ",rgp_utilities.analyzeError(np.array(diffs).T,False),xlims=xlims)
    
#------------------------------------------------------------------------------

X_base = rgp_utilities.CreateBasisVectors(XTrain, input_size, lb, ub, version="JB", num_base_vectors=num_base_vectors)
length      = ds_length[input_size]
offset      = ds_offset[input_size]    
YTst        = denormalize(YTest[:,0:1], length, offset).reshape((NTest,))    

def f(params):
    a,b = params
    model = recursiveGP(variance = a, lengthscales = b, sigma=1E-05)      
    model.initialise(X_base)
    for j in range(iterations):
        I = np.random.choice(XTrain.shape[0], batch, replace=False)      
        model.recursiveGP(XTrain[I], YTrain[I,0:1]) 
    YTst_pred   = denormalize(model.predict(XTest)[0].numpy(), length, offset).reshape((NTest,))
    return(np.max(scalar*np.abs(YTst - YTst_pred)))

    
