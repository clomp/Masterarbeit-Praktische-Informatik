# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:33:55 2022

@author: lomp
"""
__version__ = '0.1'

import os
import math
import json 
import itertools
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as PI
import gpflow
from gpflow.utilities import print_summary
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# import from this repro
import rgp_utilities 
from recursiveGP import recursiveGP
from dataset import dataset


# Constants for file operations and for the dataset meta data
DATASET_PATH    = "../../datasets/"
DATASET         = "dataset"
OUTPUT_PATH     = "output/"
META_PATH       = "meta/"
DATASET_META    = [(0.0), (6,2000),(12,2000), (12,2000), (6,None), (6,None), (9,None), (9,None)]
RANDOMTRAINING  = True

def load_dataframe(dataset_nr, loading, test_size=0.2):
    input_size, limitation = DATASET_META[dataset_nr]    
    filename        = DATASET+str(dataset_nr)
    data_filename   = DATASET_PATH+filename+".csv"
    meta_filename   = META_PATH + filename+"_meta.json"
    train_filename  = META_PATH + filename+"_train.csv"
    test_filename   = META_PATH + filename+"_test.csv"
    
    if(os.path.isfile(meta_filename) and loading):        
        TestData  = np.loadtxt(test_filename,delimiter=",")
        XTest  = TestData[ : , : input_size]
        YTest  = TestData[ : , input_size: ]
        TrainData = np.loadtxt(train_filename,delimiter=",")
        XTrain  = TrainData[ : , : input_size]
        YTrain  = TrainData[ : , input_size: ]
        f=open(meta_filename)
        ds_offset, ds_length, Lengthscales, Variances = json.load(f)
        f.close()
    else:
        dataframe = np.loadtxt(data_filename,delimiter=",")
        output_size = dataframe.shape[1]    
        ds_offset=[np.mean(dataframe[:,i:i+1]) for i in range(output_size)]
        ds_length=[np.std(dataframe[:,i:i+1]) for i in range(output_size)]      
        for i in range(output_size):
            if(ds_length[i]!=0):
                dataframe[:,i:i+1] = (dataframe[:,i:i+1]-ds_offset[i])/ds_length[i]            
            X_values    = dataframe[ : , 0 : input_size]
            Y_values    = dataframe[ : , input_size : ]    
            XTrain, XTest , YTrain, YTest = train_test_split(X_values,Y_values, test_size=test_size, random_state=42)      
            Lengthscales=[0.5]*input_size
            Variances=[1.0]*input_size
        np.savetxt(test_filename, np.hstack((XTest, YTest)), delimiter=",")
        np.savetxt(train_filename, np.hstack((XTrain, YTrain)), delimiter=",")
        out_file = open(meta_filename, "w")
        json.dump([ds_offset,ds_length,Lengthscales,Variances], out_file)
        out_file.close()   
        
    lb=[np.min(XTrain[:,i]) for i in range(input_size)]
    ub=[np.max(XTrain[:,i]) for i in range(input_size)]
    return(XTrain, XTest, YTrain, YTest, ds_offset, ds_length, lb,ub, Lengthscales, Variances)


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


dataset_nr = 7

options = {
    "num_base_vectors" : 30    ,
    "strategy" : "JB"
    }

iterations=100
batch=10
scalar=1000

num_models = 3
num_base_vectors=30

print("#### Dataset "+str(dataset_nr)+" ####")

XTrain, XTest, YTrain, YTest, ds_offset, ds_length, lb,ub, Lengthscales, Variances = load_dataframe(dataset_nr, loading=False)
input_size = DATASET_META[dataset_nr][0]

# dataobj = dataset(dataset_nr)
# dataobj.load_dataframe(reuse=False)
# XTrain = dataobj.XTrain
# YTrain = dataobj.YTrain
# XTest = dataobj.XTest
# YTest = dataobj.YTest
# Variances = dataobj.Variances
# Lengthscales = dataobj.Lengthscales
# input_size=dataobj.input_dim


print("Create recursive GPs")
modelsRGP = []
for i in range(num_models):
    #model_aux = recursiveGP(variance = dataobj.Variances[i], lengthscales = dataobj.Lengthscales[i], sigma=1E-05)
    model_aux = recursiveGP(variance = Variances[i], lengthscales = Lengthscales[i], sigma=1E-05)
    X_base = rgp_utilities.CreateBasisVectors(XTrain, input_size, version="JB", num_base_vectors=num_base_vectors)
    k=0
    while(k<100 and np.linalg.cond(model_aux.covfunc(X_base,X_base))>1E+10):
        X_base = rgp_utilities.CreateBasisVectors(XTrain, input_size, lb, ub, version="JB", num_base_vectors=num_base_vectors)
        k=k+1
    if(k>=100):
        print("Coudlnt get good base vectors for model "+str(i))
    model_aux.initialise(X_base)    
    # model_aux.initialise(dataobj, options)
    modelsRGP.append(model_aux)
    
print("Recursive training ...")    

for j in tqdm.trange(iterations):           #for j in range(iterations):               
     trainRGP(modelsRGP, j, XTrain, YTrain, batch, num_models,random=True)

#for i in tqdm.tqdm(range(0, XTrain.shape[0]+1, batch)):  
#    for j in range(num_models): 
#        modelsRGP[j].recursiveGP(XTrain[i:i+batch,:], YTrain[i:i+batch,j:j+1])

#xlims=0.1

diffs = []
NTest = YTest.shape[0]
for i in range(num_models):
    # length      = dataobj.ds_length[i+input_size]
    # offset      = dataobj.ds_offset[i+input_size]    
    length      = ds_length[i+input_size]
    offset      = ds_offset[i+input_size]        
    YTst        = denormalize(YTest[:,i:i+1], length, offset).reshape((NTest,))    
    YTst_pred   = denormalize(modelsRGP[i].predict(XTest)[0].numpy(), length, offset).reshape((NTest,))
    diffs.append(scalar*np.abs(YTst - YTst_pred))

rgp_utilities.plotting(np.array(diffs),"fullGP for "+str(dataset_nr)+" ",rgp_utilities.analyzeError(np.array(diffs).T,False),xlims=1.0)
    
