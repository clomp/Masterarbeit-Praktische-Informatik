# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:27:02 2023

@author: lomp
"""

import gpflow
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from recursiveGP import recursiveGP
from dataset import dataset

def distance(a,b):
    return(np.linalg.norm(a-b))

def get_dmin(D):
    dmin, a, b = np.linalg.norm(D[0]-D[1]), 0, 1
    for i in range(len(D)):
        for j in range(i+1,len(D)):
            d = distance(D[i], D[j])
            if(d<dmin and d!=0):
                dmin, a, b = d, i, j            
    return(dmin,a,b)


DATASET_NR=1
NUM_COORD=3
NUM_BASE_VETORS = 35

data = dataset(DATASET_NR,"full-recursive")
data.load_dataframe(reuse=False,test_size=0.99) # N = 2% of the data, dataset 1 has 10000 datapoints

#1. (Initialization) choose N data points D={(x1,y1), ..., (xN,yN)}, say N=150 and train a full GP with it.# 
print("Initialization: train "+str(NUM_COORD)+" gpflow models on 1% of data")
Dx = data.XTrain
Dy = data.YTrain[:, : NUM_COORD]

def train_fullGPs(Dx,Dy,Variances, Lengthscales, Sigma):
    kernels = [gpflow.kernels.SquaredExponential(variance=Variances[i],lengthscales=Lengthscales[i]) for i in range(NUM_COORD)]
    fullGPs = [gpflow.models.GPR(data=(Dx,Dy[:,i:i+1]),kernel=kernels[i], mean_function=None) for i in range(NUM_COORD)]

    for m in fullGPs:    
        m.likelihood.variance.assign(1E-05)
        opt    = gpflow.optimizers.Scipy()
        _      = opt.minimize(m.training_loss, m.trainable_variables)        

    Lengthscales=[ m.kernel.lengthscales.numpy() for m in fullGPs]
    Variances=[ m.kernel.variance.variables[0].value().numpy() for m in fullGPs]
    Sigma=[float(m.likelihood.variance.numpy()) for m in fullGPs]
    return(Lengthscales, Variances, Sigma)

Lengthscales, Variances, Sigma = train_fullGPs(Dx,Dy,data.Variances, data.Lengthscales, [0.1]*NUM_COORD)

#2. create Huber's recursiveGP by using the previously optimized hyperparameters of the fullGP and 
print("Create recurse GPs")
RGPs = [recursiveGP(variance = Variances[i], lengthscales = Lengthscales[i], sigma=Sigma[i]) for i in range(NUM_COORD)]

#3. choose basis vectors either from the points in D or by using Julian's method from the intervals of input coordinates of points in D.
print("Initialise each with "+str(NUM_BASE_VETORS)+" base vectors")
for m in RGPs:    
    m.initialise(data, options={"num_base_vectors" : NUM_BASE_VETORS, "strategy" : "JB"})

#4. For every new data point (x,y) 
#	(i) train the recursiveGP using Huber's algorithm 
#	(ii) substitute a point in D with (x,y) in case the "diameter" of D would get larger (in order to have greater distribution of D in the ambiente space)
#	(iii) if a certain criterion is fulfilled go back to 1. and train again a full GP with the points in D and update the hyperparameters of Huber's recursive GP.

# We use the XTest/YTest pair as a data stream:
print("Training with the initial data points")
for m in RGPs:
    j=RGPs.index(m)
    m.train_batch(data.XTrain, data.YTrain[:,j:j+1],1)

print("recursive Training ... ")
#for i in tqdm(range(len(data.XTest))):
error=[]
batch=10
dmin, a, b = get_dmin(Dx)                
dmins=[dmin]
dmin_last=dmin

N=int(data.XTest.shape[0]*0.1)

lengthscale1=[Lengthscales[0]]
for i in tqdm(range(N)):    
    x=data.XTest[i:i+1]
    y=data.YTest[i:i+1, 0:NUM_COORD]
    diff=0
    for m in RGPs:
         j=RGPs.index(m)
         yj=y[:,j:j+1]
         m.recursiveGP(x,yj)
         prediction = m.predict(x)[0]
         diff += (float(prediction-yj))**2
    error.append(diff)
    distances = [distance(Dx[i],x) for i in range(len(Dx))]    
    if(dmin+0.1 < min(distances)):
        if(distance(Dx[a],x)<distance(Dx[b],x)):
            Dx[a]=x
            Dy[a]=y            
        else:
            Dx[b]=x
            Dy[b]=y
        dmin, a, b = get_dmin(Dx)
        dmins.append(dmin)
    if(diff>0.001 or dmin>1.5*dmin_last):
        dmin_last=dmin
        print("Update of Lengthscale and Variance of full GPs")            
        Lengthscales, Variances, Sigma = train_fullGPs(Dx,Dy, Variances, Lengthscales, Sigma)                 
        lengthscale1.append(Lengthscales[0])
        for m in RGPs:
            j=RGPs.index(m)
            m.kernel = gpflow.kernels.SquaredExponential(variance=Variances[j],lengthscales=Lengthscales[j])+gpflow.kernels.White(variance=Sigma[j])
            m.covfunc = m.kernel.K    
            m.C      = m.covfunc(m.X, m.X).numpy()
            m.Kinv   = np.linalg.pinv(m.C)                                 
    
b=int(N*0.01)
Nb=int(N/b)
merr=[]
for i in range(Nb):
    merr.append(np.mean(error[b*i:b*i+b]))
     
#plt.plot(np.linspace(0,Nb,Nb), np.array(merr))
plt.plot(np.linspace(0,N,N), np.array(error))
#plt.plot(np.linspace(0,len(dmins),len(dmins)), np.array(dmins))

plt.show()
        
        
        


