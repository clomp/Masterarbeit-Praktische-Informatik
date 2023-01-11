# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:17:04 2022

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
from sklearn.cluster import KMeans

SEPERATOR = " -"*30+"\n"

def CreateBasisVectors(XTrain, input_size, version="JB", num_base_vectors=20):    
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
        num_sep = int(math.pow(num_base_vectors,1/input_size)) # take the nth root of num_base_vectors
        list_base_coordinates=[ list(np.linspace(lb[i], ub[i], num_sep))  for i in range(input_size)]
        list_base_vectors=[]
        for b in itertools.product(*list_base_coordinates):
            list_base_vectors.append(b)    
        return(np.array(list_base_vectors),lb,ub)
    elif(version=="JB"):
        list_base_coordinates=[ np.linspace(lb[i], ub[i], num_base_vectors)  for i in range(input_size)]
        for ls in list_base_coordinates:
            np.random.shuffle(ls)
        return(np.vstack(tuple(list_base_coordinates)).T, lb,ub)

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
        models[i].recursiveGP(XTrain[I], YTrain[I, i : i+1]) 

def myprint(SILENT, txt):
    if(SILENT==False):
        print(txt)


batch=10

USEHYPER=False
RANDOMTRAINING = False
BASE_VECTORS = "base_vectors.csv"
ls=4.2877551020408164
#ls=0.5 
version="JB"
num_base_vectors=100
#version="CEL"
#num_base_vectors=3**6
iterations=100
RELOAD = False 
FULLGP=True 
PLOTTING=True 
RELOAD_BASES=False
SILENT=False
    
#def testrun(ls, num_base_vectors,iterations, RELOAD = False, FULLGP=False, PLOTTING=False, RELOAD_BASES=False, SILENT=True):
#------------------------------------------------------------------------------
# load data
myprint(SILENT,SEPERATOR+"Load dataset");

#dataframe = np.loadtxt("datasets/dataset4.csv",delimiter=","); input_size=6; index_sep=6; output_size=3;  

dataframe = np.loadtxt("datasets/dataset6.csv",delimiter=","); input_size=9; index_sep = 6; output_size=6;

XTrain, XTest, YTrain, YTest = rgp_utilities.PrepareData(dataframe, input_size, output_size, index_sep,load=RELOAD, silentmode=SILENT)
X_values=np.vstack((XTrain,XTest))
Y_values=np.vstack((YTrain,YTest))

#------------------------------------------------------------------------------
myprint(SILENT,"Create Base vectors");
X_base,lb,ub = CreateBasisVectors(X_values, input_size,version=version,num_base_vectors=num_base_vectors)
num_base_vectors = X_base.shape[0]

LengthScales=[ls]*output_size  
Variances=[1.0]*output_size
Likelihoods=[9.999999974752427e-07]*output_size

if(RELOAD_BASES and os.path.isfile(BASE_VECTORS)):
    myprint(SILENT,"reload base vectors.")
    X_base = np.loadtxt(BASE_VECTORS,delimiter=",")
else:        
    np.savetxt(BASE_VECTORS,X_base,delimiter=",")    


#------------------------------------------------------------------------------
if(FULLGP):
    myprint(SILENT,SEPERATOR+"Create full GPs")    
    modelsGP = []
    for i in range(output_size):
        myprint(SILENT,"build and train "+str(i)+"th full GP with "+str(XTrain.shape[0])+" training data.")
        model_aux=gpflow.models.GPR(data=(XTrain, YTrain[:,i:i+1]), 
                    kernel=gpflow.kernels.SquaredExponential(variance=1.0,lengthscales=0.5), mean_function=None)
        model_aux.likelihood.variance.assign(1E-05)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model_aux.training_loss, model_aux.trainable_variables, options=dict(maxiter=50))        
        if(USEHYPER):
            LengthScales[i] = model_aux.kernel.lengthscales
            Variances[i]=model_aux.kernel.variance
            Likelihoods[i]=model_aux.likelihood.variance
        modelsGP.append(model_aux)     
        
#------------------------------------------------------------------------------
myprint(SILENT,SEPERATOR+"Create recursive GPs")
modelsRGP = []
for i in range(output_size):
    model_aux = recursiveGP(
        variance = Variances[i],
        lengthscales = LengthScales[i], 
        sigma = Likelihoods[i])
    model_aux.initialise(X_base)    
    modelsRGP.append(model_aux)

#------------------------------------------------------------------------------
myprint(SILENT,"Recursive training ...")    
for j in tqdm.trange(iterations):       
#for j in range(iterations):        
    trainRGP(modelsRGP, j, XTrain, YTrain, batch, output_size,random=RANDOMTRAINING)
 
    
#------------------------------------------------------------------------------
myprint(SILENT,SEPERATOR+"Results for recursive GPs:")
difference  = rgp_utilities.getError([modelsRGP[i].predict for i in range(output_size)] , XTest, YTest.T)
analysis = rgp_utilities.analyzeError(difference,SILENT)

if(FULLGP):
    myprint(SILENT,SEPERATOR+"Results for full GPs:")
    difference_fgp  = rgp_utilities.getError([modelsGP[i].predict_f for i in range(output_size)] , XTest, YTest.T)
    analysis_fgp = rgp_utilities.analyzeError(difference_fgp,SILENT)
    
#------------------------------------------------------------------------------
if(PLOTTING):
    title = "recursive GP ("+str(ls)+","+str(num_base_vectors)+","+str(iterations)+") ";
    rgp_utilities.plotting(difference.T, title ,analysis)
    if(FULLGP):        
        rgp_utilities.plotting(difference_fgp.T, "full GP",analysis_fgp)        

    #return(analysis["totalMSE"])            

  
#dataframe = np.loadtxt("datasets/dataset1.csv",delimiter=","); input_size=6; index_sep = 6; output_size=7; num_intervals = 3
#dataframe = np.loadtxt("datasets/dataset4.csv",delimiter=","); input_size=6; index_sep = 6; output_size=7; num_intervals = 3
#dataframe = np.loadtxt("datasets/dataset5.csv",delimiter=","); input_size=6; index_sep = 6; output_size=6; num_intervals = 3
#dataframe = np.loadtxt("datasets/dataset6.csv",delimiter=","); input_size=9; index_sep = 6; output_size=6; num_intervals = 2
