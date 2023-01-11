# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:39:49 2022

@author: lomp
"""
import matplotlib.pyplot as plt
import numpy as np
import gpflow


class huber():
    def __init__(self, variance=1.0 ,lengthscales=0.5, sigma = 0.1):
        self.kernel         = gpflow.kernels.SquaredExponential(variance=1.0,lengthscales=lengthscales)
        self.covfunc        = self.kernel.K
        self.mean           = gpflow.mean_functions.Zero()
        self.lengthscales   = lengthscales
        self.sn             = sigma
        self.variance       = variance
        self.noise          = lambda x : self.sn*np.random.randn()
    
    def initialise(self, lower,upper, size):        
        N = size
        self.X      = np.linspace(lower, upper, N).reshape(-1,1)
        self.mu     = self.mean(self.X).numpy()
        self.C      = self.covfunc(self.X, self.X).numpy()
        self.Kinv   = np.linalg.inv(self.C)
        self.lower = lower
        self.upper = upper
        
        
    def recursiveGP(self, xn, yn):
        Knx         = self.covfunc(xn,self.X)
        J           = Knx @ self.Kinv    
        mu_prior    = self.mean(xn) + J @ (self.mu - self.mean(self.X))                
        C_prior     = self.covfunc(xn,xn) - J @ self.covfunc(self.X, xn) + J @ self.C @  (np.transpose(J))
        G           = self.C@(np.transpose(J)) @ np.linalg.inv(C_prior + self.sn*np.eye(xn.shape[0]))    
        self.mu     = self.mu + G@(yn - mu_prior)
        self.C      = self.C - G @ J @ self.C 
        return(self.mu,self.C)
        
    def predict(self, xn):
        Knx         = self.covfunc(xn,self.X).numpy()
        Kn          = self.covfunc(xn,xn).numpy()
        mn          = self.mean(xn).numpy()
        mx          = self.mean(self.X).numpy()
        J           = Knx @ self.Kinv  
        D           = Kn - J @ Knx.T
        Cxn         = self.C @  J.T
        Cn          = D + J @ Cxn;
    
        self.mu     = mn + J @ (self.mu - mx)  
        self.C      = Cn + self.sn*np.eye(Cn.shape[0]);
        return(self.mu,self.C)
    
    def training(self, n,f,color):
        x1 = (np.random.rand(n)*(self.upper-self.lower)).reshape(-1,1)
        y1 = f(x1) + self.noise(x1)
        m1, C1 = self.recursiveGP(x1,y1)
        plt.plot(self.X, m1, 'b.',lw=0.9)        
        Y = self.predict(self.X);
        plt.plot(self.X,Y[0],color, label='Predicted Train Data')
        return(plt)
        

# latente function    
f = lambda x: np.sin(2*x)    
xp = np.linspace(0,6, 200).reshape(-1,1);
plt.plot(xp, f(xp), color="grey");


# build model
model = huber(variance=1.0, lengthscales=0.5, sigma=0.1)

# initialise model
N=30
model.initialise(0, 6, N)

model.training(20,f,'m--')
# model.training(10,f,'g--')
# model.training(30,f,'m--')


# # 1. Training  
# # x1 = (np.random.rand(40)*6).reshape(-1,1)
# # y1 = f(x1) + model.noise(x1)
# # m1, C1 = model.recursiveGP(x1,y1)
# # plt.plot(model.X, m1, 'r--')

# # X = np.linspace(0,6,N).reshape(-1,1)
# # Y_P = model.predict(X);
# # plt.plot(X,Y_P[0],'r--', label='Predicted Train Data')
# #YP_lower = np.array([YPos[i] - 1.96 * Y_P[1][i]/np.sqrt(500) for i in range(N)])
# #YP_upper = np.array([YPos[i] + 1.96 * Y_P[1][i]/np.sqrt(500) for i in range(N)])


# #   k = 2
# x2 = (np.random.rand(10)*6).reshape(-1,1)
# y2 = f(x2) + model.noise(x2)
# m2, C2 =  model.recursiveGP(x2, y2);
# plt.plot(model.X, m2, 'g--');

# Y_P = model.predict(model.X);
# plt.plot(model.X,Y_P[0],'g--', label='Predicted Train Data')


# #   k = 3
# x3 = (np.random.rand(20)*6).reshape(-1,1)
# y3 = f(x3) + model.noise(x3)
# m3, C3 = model.recursiveGP(x3, y3)
# plt.plot(model.X, m2, 'g--');

# Y_P = model.predict(model.X);
# plt.plot(model.X,Y_P[0],'m--', label='Predicted Train Data')
# plt.show()










    

