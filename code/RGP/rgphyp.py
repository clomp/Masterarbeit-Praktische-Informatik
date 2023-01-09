# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:04:18 2022

@author: lomp
"""

import matplotlib.pyplot as plt
import numpy as np
import gpflow, json, scipy
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from gpflow.utilities import print_summary
import rgp_utilities 
import tqdm as tqdm


class RGPH():
    def __init__(self, hyper=[1.0,0.01,0.5]): # hyper parameter variance, sigma, lengthscales
        self.set_hyper(hyper)   # sets variance, sigma, lengthscale
        self.covfunc        = self.get_covfunc()
        self.mean           = gpflow.mean_functions.Zero()        
        self.C_eta          = self.covfunc(self.eta,self.eta)
        self.mu_eta         = self.mean(self.eta)
        self.noise          = lambda x : self.sigma*np.random.randn()
        self.w              = 0.1
        
    def get_hyper(self,eta):    # lengthscales have been transformed via log and the inverse transformation has to be applied
        variance = np.exp(eta[0][0])
        sigma = np.exp(eta[1][0])
        if(len(eta)>3):            
            lengthscales = list(np.exp(eta[2:]).reshape((len(eta[2:]),)))            
        else:
            lengthscales=np.exp(eta[2][0])
        return(variance, sigma, lengthscales)

    def set_hyper(self, hyper):   # need to transform hyper parameters with log and exp
        self.variance = hyper[0];
        self.sigma = hyper[1];
        self.lengthscales = hyper[2];
        if(type(self.lengthscales)==list):
            
            self.eta        = np.array([np.log(self.variance), np.log(self.sigma)] +  list(np.log(self.lengthscales))).reshape(-1,1)
        else:
            self.eta        = np.array([np.log(self.variance),np.log(self.sigma), np.log(self.lengthscales)]).reshape(-1,1)                    

    def get_covfunc(self, eta = None):
        if(eta is None):
            variance, _ , lengthscales = self.get_hyper(self.eta)        
        else:
            variance, _ , lengthscales = self.get_hyper(eta)
        return(gpflow.kernels.SquaredExponential(variance = variance, lengthscales = lengthscales))                    
            
    def initialise(self, X):
        self.X      = X
        self.mu      = self.mean(self.X)
        self.C       = self.covfunc(self.X, self.X)
        self.C_g_eta = np.zeros((self.C.shape[0], self.C_eta.shape[1])) 
        self.Kinv    = np.linalg.pinv(self.C)  
        
    def inference(self, xn):    
        Knx         = self.covfunc(xn,self.X)
        Kn          = self.covfunc(xn,xn)
        J           = Knx @ self.Kinv  
        mu_prior    = J@ self.mu 
        B           = Kn - J@ Knx.T
        C_prior     = B + J @ (self.C @ J.T)
        return(mu_prior, C_prior, J)
    
    def update(self, yn, mu_prior, C_prior, J):
        G           = self.C@ J.T @ np.linalg.pinv(C_prior + self.sigma*np.eye(C_prior.shape[0]))
        self.mu     = self.mu + G@(yn - mu_prior)
        self.C      = self.C - G@J@self.C
        return(self.mu,self.C)                    
        
    def recursiveGP(self, xn, yn):
        mu_prior, C_prior, J = self.inference(xn)
        self.update(yn, mu_prior, C_prior, J)
        
    def predict(self, xn):
        mu_prior, C_prior, _ = self.inference(xn)        

        return(mu_prior,C_prior + self.sigma*np.eye(C_prior.shape[0]))
    
    def sigma_points(self):
        n=self.eta.shape[0]
        Croot = scipy.linalg.sqrtm(n*self.C_eta)
        sigmapoints=[(self.mu_eta, self.w)]
        for i in range(n):
            sigmapoints.append( (self.mu_eta + Croot[i].reshape(-1,1), (1-self.w)/(2*n)) )
        for i in range(n):
            sigmapoints.append( (self.mu_eta - Croot[i].reshape(-1,1), (1-self.w)/(2*n)) )
        return(sigmapoints)
    
    
    def Equations1415(self, xn,eta):
        n1 = self.X.shape[0]
        n2 = self.eta.shape[0]
        n3 = xn.shape[0]
        
        
        K = self.get_covfunc(eta)   # get kernel function wrt eta
        Knx = K(xn,self.X)
        Kxx = K(xn,xn)
        Kinv = np.linalg.pinv(K(self.X,self.X))
        
        J   = Knx @ Kinv  
        A_eta = np.vstack((np.eye(n1+n2),np.hstack((J,np.zeros((n3,n2))))))
        
        b = self.mean(xn) - J@self.mean(self.X)
        mu_w_eta = np.vstack((np.zeros((n1+n2,1)),b))
        B_eta = Kxx - J@(Knx.T)
        C_w_eta = np.vstack((np.zeros((n1+n2,n1+n2+n3)),np.hstack((np.zeros((n3,n1+n2)),B_eta))))
        
        return(A_eta, mu_w_eta, C_w_eta)
    
    def Equations1617(self,xn):
        n1 = self.X.shape[0]
        n2 = self.eta.shape[0]
        sigmapoints=self.sigma_points()
        S = self.C_g_eta @ np.linalg.pinv(self.C_eta)
        aux=[]
        mu_p_t=None
        for eta_hat, weight_i in sigmapoints:
            A_eta, mu_w_eta, C_w_eta = self.Equations1415(xn,eta_hat)
            mu_p_i = A_eta @ np.vstack( (self.mu + S@(eta_hat - self.mu_eta), eta_hat) ) + mu_w_eta
            if(mu_p_t is None):
                mu_p_t = weight_i*mu_p_i 
            else:
                mu_p_t += weight_i*mu_p_i 
            H = np.vstack( (np.hstack((self.C - S@(self.C_g_eta.T),np.zeros((n1,n2)))),np.zeros((n2,n1+n2))))
            C_p_i = A_eta@ H @(A_eta.T) + C_w_eta
            aux.append( (weight_i, mu_p_i, C_p_i) )
        C_p_t = sum( [ w * ( (mu - mu_p_t)@(mu - mu_p_t).T + C) for w,mu,C in aux])   
        return(mu_p_t, C_p_t) 
    
    
#------------------------------------------------------------------------------
#dataset="dataset6" ; input_size=9; limitation=None;
dataset="dataset5" ; input_size=6; limitation=None;
output_size=3
num_base_vectors=20
iterations=1
batch=10
scalar=1000
xlims=1.0
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
#------------------------------------------------------------------------------
def denormalize(A,length,offset):
    return((A*length)+offset)

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
#------------------------------------------------------------------------------
lb=[np.min(XTrain[:,i]) for i in range(input_size)]
ub=[np.max(XTrain[:,i]) for i in range(input_size)] 

print("Create recursive GPs")
modelsRGP = []
for i in range(output_size):
    hyper = [Variances[i], 1E-05, Lengthscales[i]]
    model_aux = RGPH(hyper)      
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
    trainRGP(modelsRGP, j, XTrain, YTrain, batch, output_size,random=True)


diffs = []
NTest = YTest.shape[0]
for i in range(output_size):
    length      = ds_length[i+input_size]
    offset      = ds_offset[i+input_size]    
    YTst        = denormalize(YTest[:,i:i+1], length, offset).reshape((NTest,))    
    YTst_pred   = denormalize(modelsRGP[i].predict(XTest)[0].numpy(), length, offset).reshape((NTest,))
    diffs.append(scalar*np.abs(YTst - YTst_pred))

rgp_utilities.plotting(np.array(diffs),"fullGP "+dataset+" ",rgp_utilities.analyzeError(np.array(diffs).T,False),xlims=xlims)
    
m=model_aux
SP=m.sigma_points()
eta=SP[1][0]
xn=XTrain[0:1]
mu_p_t, C_p_t =  m.Equations1617(xn)
