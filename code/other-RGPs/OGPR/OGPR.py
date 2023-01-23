# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:45:41 2022

"On the relationship between online Gaussian process regresion and KLMS algorithms"
Steven Van Vaerenbergh, et. al. (2016)

@author: lomp
"""

import matplotlib.pyplot as plt
import numpy as np
import gpflow, json
from gpflow.utilities import print_summary
import rgp_utilities 
import tqdm as tqdm
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def embedding(A):
    n,m=A.shape
    return(np.hstack((np.vstack((A,np.zeros((1,m)))),np.zeros((n+1,1)))))

class OGPR():
    def __init__(self, variance=1.0 ,lengthscales=0.5, sigma = 0.01):
        self.kernel   = gpflow.kernels.SquaredExponential(variance=variance,lengthscales=lengthscales)+gpflow.kernels.White(variance=sigma)
        self.K        = self.kernel.K
        self.sn       = sigma        
    
    def init(self, D):
        self.X      = D[0]
        self.y      = D[1]
        self.mu   = gpflow.mean_functions.Zero()(self.X)
        self.Sigma  = self.K(self.X ,self.X )
        self.Q      = np.linalg.pinv(self.Sigma)
        
    def update(self, x, y):
        position=np.where((self.X == x).all(axis=1))[0]
        if( len(position) == 0):
            q = self.Q @ self.K(self.X,x)
            h = self.Sigma @ q
            gamma = self.K(x,x) - self.K(x,self.X) @ self.Q @ self.K(self.X,x)        
            sigma_f = gamma + q.T @ h        
            sigma_y = self.sn + sigma_f        
            y_hat = q.T@self.mu        
            qe = np.vstack((q,-np.ones(1)))        
            Q_neu = embedding(self.Q) + (1/gamma)*(qe@qe.T)        
            he = np.vstack((h,sigma_f))        
            mu_neu = np.vstack((self.mu,y_hat)) + (y-y_hat)/sigma_y * he        
            Sigma_neu = np.hstack((np.vstack((self.Sigma, h.T)), he)) - 1/sigma_y * (he@he.T)        
            self.X = np.vstack((self.X,x))
            self.y = np.vstack((self.y,y))
            self.mu = mu_neu
            self.Sigma = Sigma_neu
            self.Q = Q_neu
            return(-1)
        else:
            return(position[0])
    
    def predict(self, x):
        KXx = self.K(self.X,x)
        mu = KXx.T @ self.Q @ self.mu
        sigma = self.K(x,x) - (KXx.T)@self.Q@KXx
        return(mu,sigma)


def denormalize(A,length,offset):
    return((A*length)+offset)

dataset="dataset4" ; input_size=6; limitation=None;
#dataset="dataset5" ; input_size=6; limitation=None;
#dataset="dataset6" ; input_size=9; limitation=None;

output_bit=0
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

Y=YTrain[:,output_bit:output_bit+1]

model = OGPR()
model.init((XTrain[0:1],Y[0]))
print("Training")
for i in range(1,XTrain.shape[0]):
    model.update(XTrain[i:i+1],Y[i])

print("Testing")
length=ds_length[input_size+output_bit]
offset=ds_offset[input_size+output_bit]
Yd = denormalize(YTest[:,output_bit:output_bit+1],length,offset)
Yp = denormalize(model.predict(XTest)[0],length,offset)
print("Mean:"+str(np.mean(1000*np.abs(Yd-Yp))))
print("Max:"+str(np.max(1000*np.abs(Yd-Yp))))


    