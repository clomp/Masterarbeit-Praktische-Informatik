# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:48:47 2022
@author: lomp

Based on "Local Gaussian Process Regression for Real Time Online Model Learning and Control"
https://dl.acm.org/doi/10.5555/2981780.2981929

doi
"""
import gpflow
import numpy as np
import json
import tqdm as tqdm
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from dataset import dataset


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

    def train_batch(self, xt, yt, batch=None):            # batch wird ignoriert
        for i in tqdm.tqdm(range(xt.shape[0])):  
            self.update(xt[i], yt[i])
    
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
            
    def predict(self, X):
        P=np.array([self.predict_mean(X[i]) for i in range(len(X))]).reshape((-1,1))
        return(P,P)
        
    def predict_mean(self, x):
        N=len(self.center_of_models)
        ws = [self.distance(x,self.center_of_models[k]) for k in range(N)]
        ys = [ (self.K(self.local_models_X[k], x.reshape(1,-1)).T) @ self.alpha[k] for k in range(N)]
        mean = sum(ws)
        return(1/mean * sum([ws[k]*ys[k] for k in range(N)]))
    
 
dataobj = dataset(6)
dataobj.load_dataframe(reuse=True)

XTest  = dataobj.XTest
YTest  = dataobj.YTest
XTrain  = dataobj.XTrain
YTrain  = dataobj.YTrain
ds_offset = dataobj.ds_offset
ds_length = dataobj.ds_length
Lengthscales = dataobj.Lengthscales
Variances = dataobj.Variances
           
num_models=3
SCALING = 1000
print("Create Models")
models = [ localGPR(dataobj.XTrain[0],dataobj.YTrain[0,i:i+1], 0.1,1.0,1.0,0.9) for i in range(num_models)]


classname=localGPR.__name__
print("Training of ", num_models, " models of class ", classname)
for m in models:
   j=models.index(m)
   print("Training of "+str(j)+"th model")    
   m.train_batch(dataobj.XTrain, dataobj.YTrain[:,j:j+1])

# print("Training")
# for m in models:
#     j=models.index(m)
#     print("Training of "+str(j)+"th model")
#     for i in tqdm.trange(1,len(XTrain)):
#         m.update(XTrain[i],YTrain[i,j:j+1])
                 
# predictions = []
# for m in models:
#     j = models.index(m)
#     print("Prediction of "+str(j)+"th model")
#     predictions.append([np.abs(dataobj.YTest[i,j:j+1]-m.predict(dataobj.XTest[i])) for i in range(len(dataobj.YTest))])

# # print("Max: " + str(np.max(diff)))
# # print("Mean: " + str(np.mean(diff)))
    

# print("Calculate Predictions and analyze error")   
# predictions = [m.predict(dataobj.XTest[i]).numpy() for i in range(len(XTest))]
# difference, totalMSE, componentwiseErrors = dataobj.analyze_error(predictions,SCALING)
    
# print("Print error analysis")   
# dataobj.print_analysis(difference.T, totalMSE, componentwiseErrors)  