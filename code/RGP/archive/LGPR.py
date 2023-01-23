# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:11:08 2022

@author: lomp
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
from matplotlib import pyplot as plt
import gpy




# the function f
#def f(x):
#    return(np.sin(2*np.pi*x))

def f(x):
    return(np.sin(4*x))

# the distance function with W the lengthscales
def dist(x,c,W):
    return(np.exp(-0.5*(np.transpose(x-c)@W@(x-c))))

Interval = np.linspace(-1,1,5000)
Interval2 = np.linspace(-1,1,100)

# center of local models (equidistance) on [-1,1]
num_models = 10
centers  = [-1.0 + (2.0/num_models)*i for i in range(num_models)]
centers_np  = np.array([centers]).T

# create GP model on centers
  
model   = gpy.models.GPRegression(centers_np, 
                                  f(centers_np), 
                                  kernel=gpy.kern.RBF(input_dim=1, variance=1., lengthscale=.9))
    
#Train Model
model.optimize_restarts(num_restarts=1)
model.optimize

# info about the model
kern    = model.kern.K
W       = np.eye(1)*model.kern.lengthscale/10
sigma   = model.Gaussian_noise.variance

# initialize local models
local_X = [ [centers[i]]           for i in range(num_models)]
local_Y = [ [f(centers[i])]        for i in range(num_models)]
local_K = [ kern(np.array([[centers[i]]]),np.array([[centers[i]]])) for i in range(num_models)]
local_alpha = [np.array((1/local_K[i])*local_Y[i]) for i in range(num_models) ]

threshold = 0.9


def training(x,y):

    # find the center that is closest to x
    k = 0
    v = dist(x,np.array([centers[0]]),W) 
    num_models = len(centers)
    for i in range(1,num_models):
        w=dist(x,np.array([centers[i]]),W)
        if (v < w):
            v=w
            k=i
    
    if(v > threshold):
        # insert (x,y) to the nearest local model
        if(x not in local_X[k]):
            local_X[k].append(x)
            locX = np.array([np.array(local_X[k])]).T
            local_Y[k].append(y)
            locY = np.array(local_Y[k]).T
            locY=locY.reshape(-1,1)
            centers[k] = np.mean(local_X[k])              
            local_alpha[k] = np.linalg.pinv( kern(locX,locX) + sigma*np.eye(len(local_X[k])))@locY
    else:
        # create a new model#
        centers.append(x)        
        local_X.append([x])
        local_Y.append([y])   
        local_alpha.append((1/kern(np.array([[x]]),np.array([[x]])))*y)
        
        
def predict(x):
    num_models = len(centers)
    wk = []
    yk = []
    for i in range(num_models):
        xx  = np.array([x])
        cc  = np.array([[centers[i]]])
        wk.append(dist(xx,cc,W))
        locX = np.array([np.array(local_X[i])]).T
        kk  = kern(locX, xx)
        yk.append(kk.T@local_alpha[i])
    y = sum(wk[i]*yk[i] for i in range(num_models))/sum(wk)
    return(y)

def batch(n):
    for i in range(n):
        for c in centers:
            training(c,f(c))
            
batch(2)

X = [-1 + 2/100 * i for i in range(500)]
Y=f(X)

for x in X:
    training(x,f(x))
    
#Y,_=model.predict(centers)
Xp= np.linspace(-1,1,100).reshape(-1,1)
Yp = np.array([predict(x) for x in Xp]).reshape(-1,1)

plt.plot(Interval, f(Interval))
#plt.scatter(centers,Y,color="green",lw=1)
plt.plot(Xp,Yp,"r+")
plt.show()


