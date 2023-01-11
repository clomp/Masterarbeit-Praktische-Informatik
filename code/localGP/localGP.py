# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:48:47 2022

@author: lomp
"""
import gpflow
import numpy as np
import json
import tqdm as tqdm
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class localGPR():
    def __init__(self, x, y, sigma_n, sigma_s, lengthscales, wgen=0.1):    
        self.sigma = sigma_n
        if(type(lengthscales) is list):
            if(len(lengthscales) == x.shape[0]): 
                self.W = np.diag(lengthscales)
            else:
                print("Error: size of lengthscales does not match data vectors.")
        else:
            self.W = lengthscales * np.eye(x.shape[0])
        self.wgen = wgen
        self.kernel = gpflow.kernels.SquaredExponential(variance=sigma_s,lengthscales=lengthscales)
        self.K = self.kernel.K
        self.center_of_models = [x]
        self.local_models_X = [[x]]
        self.local_models_y = [[y]]
        self.alpha = [self.AdjustLocalPredictionVector(0)]
        
    def AdjustLocalPredictionVector(self, k):
        X=self.local_models_X[k]
        y=self.local_models_y[k]
        C=self.K(X,X)
        return((np.linalg.pinv(C+self.sigma*np.eye(len(X)))@y).reshape(-1,1))
    
    def distance(self, x1,x2):
        d=x1-x2
        return(np.exp(-0.5*d @ self.W @ (d.T)))
                
    def update(self, x, y):
        distances = [self.distance(x,cx) for cx in self.center_of_models]
        k=np.argmax(distances)
        if (distances[k] > self.wgen):
            if(list(x) not in [list(a) for a in self.local_models_X[k]]):
                self.local_models_X[k].append(x)
                self.local_models_y[k].append(y)            
                self.center_of_models[k]=np.mean(self.local_models_X[k], axis=0)
                self.alpha[k] = self.AdjustLocalPredictionVector(k)                
            else:
                return(False)
        else:
            self.center_of_models.append(x)
            self.local_models_X.append([x])
            self.local_models_y.append([y])            
            self.alpha.append(self.AdjustLocalPredictionVector(len(self.center_of_models)-1))
            
        
    def predict(self, x):
        N=len(self.center_of_models)
        ws = [self.distance(x,self.center_of_models[k]) for k in range(N)]
        ys = [ (self.K(self.local_models_X[k], x.reshape(1,-1)).T) @ self.alpha[k] for k in range(N)]
        mean = sum(ws)
        return(1/mean * sum([ws[k]*ys[k] for k in range(N)]))
    
    
DATA_PATH="../../dataset/"
#dataset="dataset4" ; input_size=6; limitation=None;
#dataset="dataset5" ; input_size=6; limitation=None;
dataset="dataset6" ; input_size=9; limitation=None;
#dataset="dataset7" ; input_size=9; limitation=None;
#dataset="dataset1" ; input_size=6; limitation=2000;
#dataset="dataset2" ; input_size=12; limitation=2000;
#dataset="dataset3" ; input_size=12; limitation=2000;

TEST_DATA = "Test_Data_"
TRAIN_DATA = "Train_Data_"
META_DATA = "Meta_Data_"

TestData  = np.loadtxt(TEST_DATA+dataset+".csv",delimiter=",")
XTest  = TestData[ : , : input_size]
YTest  = TestData[ : , input_size: ]
TrainData = np.loadtxt(TRAIN_DATA+dataset+".csv",delimiter=",")
XTrain  = TrainData[ : , : input_size]
YTrain  = TrainData[ : , input_size: ]
f=open(META_DATA+dataset+".json")
ds_offset, ds_length, Lengthscales, Variances = json.load(f)
f.close()    
    
            
m=localGPR(XTrain[0],YTrain[0,0], 0.1,1.0,1.0,0.9)

print("Training")
for i in range(1,len(XTrain)):
    m.update(XTrain[i],YTrain[i,0])

diff=[]
for i in tqdm.trange(100):
    diff.append(np.abs(YTest[i,0]-m.predict(XTest[i])))

print("Max: " + str(np.max(diff)))
print("Mean: " + str(np.mean(diff)))
    
