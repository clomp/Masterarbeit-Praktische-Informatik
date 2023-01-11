# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:17:04 2022

@author: lomp
"""


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
from sklearn.cluster import KMeans


def PrepareData(dataframe,input_size, output_size, index_sep, test_size_proportion=0.2):

    # normalize data to obtain zero mean and std 1
    num_data = dataframe.shape[1]
    ds_offset=[np.mean(dataframe[:,i:i+1]) for i in range(num_data)]
    ds_length=[np.std(dataframe[:,i:i+1]) for i in range(num_data)]  
    for i in range(num_data):
        if(ds_length[i]!=0):
            dataframe[:,i:i+1] = (dataframe[:,i:i+1]-ds_offset[i])/ds_length[i] 

    # split data in input and output
    X_values    = dataframe[ : , 0 : input_size]
    Y_values    = dataframe[ : , index_sep : index_sep + output_size]
    
            
    # splitt data into train and test ets
    XTrain, XTest , YTrain, YTest = train_test_split(X_values,Y_values, test_size=test_size_proportion, random_state=42)
    
    return(X_values, Y_values, XTrain, XTest, YTrain, YTest)

def CreateBasisVectors(XTrain, input_size, version="kmeans", num_base_vectors=20):    
    lb=[np.min(XTrain[:,i]) for i in range(input_size)]
    ub=[np.max(XTrain[:,i]) for i in range(input_size)] 
    N=len(XTrain)
    if(version=="kmeans"):
        kmeans = KMeans(n_clusters=num_base_vectors, random_state=42)    
        kmeans.fit(XTrain)
        I=[[j for j in range(N) if kmeans.labels_[j]==i] for i in range(num_base_vectors)]
        Cent=[[np.mean(XTrain[I[i]][:,j]) for j in range(input_size)] for i in range(num_base_vectors)]
        return(np.array(Cent),lb,ub)
    elif(version=="CEL"):        
        list_base_coordinates=[ list(np.linspace(lb[i], ub[i], num_base_vectors))  for i in range(input_size)]
        list_base_vectors=[]
        for b in itertools.product(*list_base_coordinates):
            list_base_vectors.append(b)    
        return(np.array(list_base_vectors),lb,ub)
    elif(version=="JB"):
        list_base_coordinates=[ np.linspace(lb[i], ub[i], num_base_vectors)  for i in range(input_size)]
        for ls in list_base_coordinates:
            np.random.shuffle(ls)
        return(np.vstack(tuple(list_base_coordinates)).T, lb,ub)

def plotting(evaluation, label, scalar=1000, nbins=100,xlims=5):
    output_size=len(evaluation)
    figure, ax = plt.subplots(1,output_size, figsize = (10,10))    
    figure.suptitle(label)
    for i in range(output_size):
        ax[i].hist(scalar*evaluation[i],bins=nbins)    
        ax[i].set_xlim([0, xlims])    
    plt.show()    
    

def comparePrediction(predict_fns, XTest, YTest):
    Ntest = XTest.shape[0]
    output_size=YTest.shape[0]
    evaluation = np.empty((output_size,Ntest))
    for i in range(output_size): 
        Mean,_  = predict_fns[i](XTest)    
        Y_pred = Mean.numpy().reshape((Ntest,))                
        evaluation[i] = np.abs(Y_pred - YTest[i])
    return(evaluation)

def MeanNorm(D):
    return(np.mean([np.linalg.norm(x) for x in D]))

def NLL(M,C,Y):
    D=np.diag(C)
    s= sum(np.log(D[i]) + 1/D[i] * (Y[i]-M[i])**2  for i in range(len(D)))
    return(0.5*s)


#def testrun(num_intervals,  ls=4.17, iterations=100, version="JB", description=False, FULLGP=False, PLOTTING=False):

num_base_vectors=200
ls=4.1466;
iterations=100;
#version="kmeans";
version="JB";
description=False;
FULLGP=False; 
PLOTTING=False;

if(True):    
    #dataframe = np.loadtxt("datasets/dataset1.csv",delimiter=","); input_size=6; index_sep = 6; output_size=7; num_intervals = 3
    #dataframe = np.loadtxt("datasets/dataset4.csv",delimiter=","); input_size=6; index_sep = 6; output_size=7; num_intervals = 3
    #dataframe = np.loadtxt("datasets/dataset5.csv",delimiter=","); input_size=6; index_sep = 6; output_size=6; num_intervals = 3
    #dataframe = np.loadtxt("datasets/dataset6.csv",delimiter=","); input_size=9; index_sep = 6; output_size=6; num_intervals = 2

    dataframe = np.loadtxt("datasets/dataset4.csv",delimiter=","); input_size=6; index_sep=6; output_size=3;  

    batch=10
    X_values, Y_values, XTrain, XTest, YTrain, YTest = PrepareData(dataframe, input_size, output_size, index_sep)

    num_dataset = X_values.shape[0]

    X_base,lb,ub = CreateBasisVectors(X_values, input_size,version=version,num_base_vectors=num_base_vectors)
    #LengthScales = list(2*(np.array(ub)-np.array(lb))/num_intervals)        
    #X_base,lb,ub, LengthScales = CreateBasisVectors(X_values, input_size,version="CEL", num_intervals=3)
    
    LengthScales=[ls,ls,ls]
    
    num_base_vectors = X_base.shape[0]


    modelsRGP = []

    for i in range(output_size):
        model_aux = recursiveGP(variance = 1.0, lengthscales = LengthScales[i], sigma = 0.0)
        model_aux.initialise(X_base)    
        modelsRGP.append(model_aux)

    for j in tqdm.trange(iterations):
#    for j in range(iterations):        
        I = np.random.choice(num_dataset, batch, replace=False)  
        for i in range(output_size): 
            modelsRGP[i].recursiveGP(X_values[I], Y_values[I, i : i+1]) 

    print("Results:")
    
    difference  = comparePrediction(
        [modelsRGP[i].predict for i in range(output_size)] , 
        XTest, 
        np.transpose(YTest))

    mn=MeanNorm(np.transpose(difference))
    print("Mean Norm:"+str(mn))
    if(description):
        for i in range(output_size):
            mean, covariance = modelsRGP[i].predict(XTest)
            print("NLL "+str(i)+":"+str(NLL(mean.numpy(),covariance, YTest.T[i])))
            
    if(FULLGP):
        print("Create full GP")    
        modelsGP = []
        for i in range(output_size):
            print("build and train "+str(i)+"th full GP with "+str(num_base_vectors)+" training data.")
            model_aux=gpflow.models.GPR(data=(XTrain[:num_base_vectors], YTrain[:num_base_vectors,i:i+1]), 
                        kernel=gpflow.kernels.SquaredExponential(variance=1.0,lengthscales=list((np.array(ub)-np.array(lb))/2)), 
                        mean_function=None)
            model_aux.likelihood.variance.assign(1E-05)
            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(model_aux.training_loss, model_aux.trainable_variables)
            #print(model_aux.kernel.lengthscales)
            modelsGP.append(model_aux)

        print("Evaluation:")

        difference_fgp = comparePrediction([modelsGP[i].predict_f for i in range(output_size)] , XTest, np.transpose(YTest))

        print("Mean Square sum:"+str(MeanNorm(np.transpose(difference_fgp))))
        print("NLL for full GP:")
        for i in range(output_size):
            mean, covariance = modelsGP[i].predict_f(XTest)
            print("NLL "+str(i)+":"+str(NLL(mean.numpy(),covariance, YTest.T[i])))

    if(PLOTTING):
        plotting(difference, "xyz-errors in mm for RGP")
        plotting(difference_fgp,"xyz-errors in mm for full GP")
        



