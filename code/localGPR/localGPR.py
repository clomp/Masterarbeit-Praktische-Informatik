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
import tqdm as tqdm
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class localGPR():
    def __init__(self, variance, lengthscales, sigma):    
        self.sigma = sigma        
        self.kernel = gpflow.kernels.SquaredExponential(variance=variance,lengthscales=lengthscales)
        self.K = self.kernel.K        
        self.lengthscales=lengthscales
                
    def initialise(self, dataobj, options=None):
        x = dataobj.XTrain[0]        
        y = dataobj.YTrain[0,options["i"]]
        self.center_of_models = [x]
        self.local_models_X = [[x]]
        self.local_models_y = [[y]]
        self.alpha = [self.AdjustLocalPredictionVector(0)]
        self.wgen = options["wgen"]                
        if(type(self.lengthscales) is list):
            if(len(self.lengthscales) == x.shape[0]): 
                self.W = np.diag(self.lengthscales)
            else:
                print("Error: size of lengthscales does not match data vectors.")
        else:
            self.W = self.lengthscales * np.eye(x.shape[0])            
        
    def AdjustLocalPredictionVector(self, k):
        X=self.local_models_X[k]
        y=self.local_models_y[k]
        C=self.K(X,X)
        return((np.linalg.pinv(C+self.sigma*np.eye(len(X)))@y).reshape(-1,1))
    
    def distance(self, x1,x2):
        d=x1-x2
        return(np.exp(-0.5*d @ self.W @ (d.T)))

    def train_batch(self, xt, yt, batch=None):            # batch wird ignoriert
        yt=np.squeeze(yt)
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
        P=[]
        for i in tqdm.trange(len(X)):
            P.append(self.predict_mean(X[i]))             
        P=np.array(P).reshape((-1,1))
        return(P,None) 
        
    def predict_mean(self, x):
        N=len(self.center_of_models)
        ws = [self.distance(x,self.center_of_models[k]) for k in range(N)]
        ys = [ (self.K(self.local_models_X[k], x.reshape(1,-1)).T) @ self.alpha[k] for k in range(N)]
        mean = sum(ws)
        return(1/mean * sum([ws[k]*ys[k] for k in range(N)]))
    
 
# dataobj = dataset(6)
# dataobj.load_dataframe(reuse=True)
# num_coordinates=3
# SCALING = 1000

# #rint("Create Models")
# #odels = [ localGPR(dataobj.XTrain[0],dataobj.YTrain[0,i:i+1], 0.1,1.0,1.0,0.9) for i in range(num_coordinates)]

# rgpmodel  = localGPR 
# options = {"wgen":0.5}

# classname = rgpmodel.__name__
# print("Create ", num_coordinates, " models of class ", classname)
# models = [rgpmodel(variance = dataobj.Variances[i], lengthscales = dataobj.Lengthscales[i], sigma=1E-05) for i in range(num_coordinates)]


# print("Initialitze ", num_coordinates, " models of class ", classname)
# for m in models:
#     options["i"]=models.index(m)
#     m.initialise(dataobj, options)

# print("Training of ", num_coordinates, " models of class ", classname)
# for m in models:
#     j=models.index(m)
#     m.train_batch(dataobj.XTrain, dataobj.YTrain[:,j:j+1],None)

# print("Calculate Predictions and analyze error")   
# predictions = [m.predict(dataobj.XTest)[0] for m in models]
# difference, totalMSE, componentwiseErrors = dataobj.analyze_error(predictions,SCALING)
    
# print("Print error analysis")   
# dataobj.print_analysis(difference.T, totalMSE, componentwiseErrors)  

