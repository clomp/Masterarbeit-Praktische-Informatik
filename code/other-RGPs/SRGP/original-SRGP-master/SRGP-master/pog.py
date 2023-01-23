# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:13:31 2022

@author: lomp
"""

import numpy as np
import json,os,math
# loading----------------------------------------------------------------------
TEST_DATA = "Test_Data_"
TRAIN_DATA = "Train_Data_"
META_DATA = "Meta_Data_"
#dataset="dataset4" ; input_size=6; limitation=None;
#dataset="dataset5" ; input_size=6; limitation=None;
#dataset="dataset6" ; input_size=9; limitation=None;
dataset="dataset7" ; input_size=9; limitation=None;
#dataset="dataset1" ; input_size=6; limitation=2000;
#dataset="dataset2" ; input_size=12; limitation=2000;
#dataset="dataset3" ; input_size=12; limitation=2000;
output_size=3
iterations=100
batch=10
scalar=1000
print("#### "+dataset+" ####")
TestData  = np.loadtxt(TEST_DATA+dataset+".csv",delimiter=",")
XTest  = TestData[ : , : input_size]
YTest  = TestData[ : , input_size: ]
TrainData = np.loadtxt(TRAIN_DATA+dataset+".csv",delimiter=",")
XTrain  = TrainData[ : , : input_size]
YTrain  = TrainData[ : , input_size: ]
f=open(META_DATA+dataset+".json")
ds_offset, ds_length, Lengthscales, Variances = json.load(f)
f.close()    

# koppel ----------------------------------------------------------------------
c=1.0


def kernel(x1,x2):
    return(np.exp( (-1/c)*(x1-x2)@(x1-x2).T))

def Kx(S,x):
    L=[0.0]*len(S)
    for i in range(len(S)):
        L[i]=kernel(S[i],x)
    return(np.array([L]))

def K(S):
    L=np.array([[0.0]*len(S)]*len(S))
    for i in range(len(S)):        
        L[i,i]=1.0
        for j in range(i+1,len(S)):
            h=kernel(S[i],S[j])            
            L[i,j]=h
            L[j,i]=h
    return(L)

def predict(D,Y,x,sigma=1E-05):            
    KDx   = Kx(D,x)
    Kinv  = np.linalg.pinv(K(D)+sigma*np.eye(len(D)))
    mu    = KDx @ Kinv @ Y
    Sigma = K(x,x) - KDx @ Kinv @ (KDx.T) + sigma 
    return(mu,Sigma)

def POG(stream, sigma=1E-05):
    S=np.array([])
    Y=np.array([])
    for (x,y,epsilon) in stream:
        if(S.size==0):
            S=np.array([x])            
            Y=np.array([y])                        
        else:
            mu, Sigma = predict(S,Y,x,sigma)
            Stilde = np.vstack((S,np.array([x])))
            Ytilde = np.vstack((Y,np.array([y])))
            S, Y = DHMP(mu, Sigma, Stilde,Ytilde, epsilon)
    
def Hellinger( mu_Sigma1, mu_Sigma2):    
    mu1, Sigma1 = mu_Sigma1
    mu2, Sigma2 = mu_Sigma2
    Sigmab     = 0.5*(Sigma1+Sigma2)
    det_Sigma1 = np.linalg.det(Sigma1)
    det_Sigma2 = np.linalg.det(Sigma2)
    det_Sigmab = np.linalg.det(Sigmab)
    dets    = (det_Sigma1 * det_Sigma2)**(1/4)*(det_Sigmab**(-1/2))
    u       = mu1 - mu2
    return(np.sqrt(1-dets*np.exp(-(1/8)*(u.T@np.linalg.pinv(Sigmab)@u))))
    
def DHMP(mu, Sigma, D, Y, epsilon):
    M=len(D)
    I=list(range(M))
#    while(len(I!=0)):
    distances=[Hellinger((mu,Sigma),predict(np.delete(D,(j),axis=0),np.delete(Y,(j),axis=0),D[j])) for j in range(len(D))]
    return(distances)    
            
    
D=XTrain[:3]
Y=YTrain[:3,0:1]
x=XTrain[3]
y=YTrain[3,0]
predict(D,Y,x)            
        
