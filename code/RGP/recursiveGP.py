# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:11:46 2022

@author: lomp
"""

import matplotlib.pyplot as plt
import numpy as np
import gpflow
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from gpflow.utilities import print_summary

class recursiveGP():
    def __init__(self, variance=1.0 ,lengthscales=0.5, sigma = 0.01):
        self.kernel         = gpflow.kernels.SquaredExponential(variance=variance,lengthscales=lengthscales)+gpflow.kernels.White(variance=sigma)
        self.covfunc        = self.kernel.K
        self.mean           = gpflow.mean_functions.Zero()
        self.sn             = sigma        
        self.noise          = lambda x : self.sn*np.random.randn()
            
    def initialise(self, X):
        self.X      = X
        self.mu     = self.mean(self.X).numpy()
        self.C      = self.covfunc(self.X, self.X).numpy()
        self.Kinv   = np.linalg.pinv(self.C)       # in cae C is singular
        
    def estimate_hyper(self, X,Y):
        model_estimate = gpflow.models.GPR(data=(X, Y), 
                        kernel=self.kernel, 
                        mean_function=self.mean)        
        model_estimate.likelihood.variance.assign(self.sn)        
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model_estimate.training_loss, model_estimate.trainable_variables)        
        print_summary(model_estimate)                
        self.sn = model_estimate.likelihood.variance
        return(model_estimate)
    

    def estimate_hyper_line(self, f, lower, upper, N):
        I=np.linspace(lower, upper, 100*N)
        X=np.random.choice(I,N).reshape(-1,1)
        Y=f(X)
        model_estimate = gpflow.models.GPR(data=(X, Y), 
                        kernel=self.kernel, 
                        mean_function=self.mean)
        model_estimate.likelihood.variance.assign(self.sn)
        
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model_estimate.training_loss, model_estimate.trainable_variables)
        print_summary(model_estimate)                
        self.sn = model_estimate.likelihood.variance        
        #print("Variance (estd)):"+str(self.variance))
        #print("Lengthscales (estd)):"+str(self.lengthscales))        
        #print("Sigma (estd)):"+str(self.sn))        
        return(model_estimate)
    
    
        
    def recursiveGP(self, xn, yn):
        Knx         = self.covfunc(xn,self.X).numpy()
        Kn          = self.covfunc(xn,xn).numpy()
        mn          = self.mean(xn) 
        mx          = self.mean(self.X)        
        J           = Knx @ self.Kinv  
#        J           = Knx @ np.linalg.pinv(self.C)
        D           = Kn - J @ Knx.T        
        # covariance of new data
        Cxn         = self.C @ J.T
        Cn          = D + J @ Cxn
        # mean of new data
        mn          = mn + J @ (self.mu - mx)        
        # Kalman Gain
        G = Cxn @ np.linalg.pinv(Cn + self.sn * np.eye(yn.shape[0]))      # to prevent singularity
        # Update basis vectors
        self.mu     = self.mu  + G@(yn - mn)
        self.C      = self.C - G@Cxn.T    
        return(self.mu,self.C)
        
    def predict(self, xn):
        Knx         = self.covfunc(xn,self.X).numpy()
        Kn          = self.covfunc(xn,xn).numpy()
        mn          = self.mean(xn)
        mx          = self.mean(self.X)        
        J           = Knx @ self.Kinv  
#       J           = Knx @ np.linalg.pinv(self.C)
        D           = Kn - J @ Knx.T
        # covariance of new data
        Cxn         = self.C @  J.T
        Cn          = D + J @ Cxn;
        # mean of new data
        mu_pred     = mn + J @ (self.mu - mx)        
        C_pred      = Cn + self.sn*np.eye(Cn.shape[0]);
        return(mu_pred,C_pred)
        
    def lossRMSE(self, Y,Yp):
        return(np.sqrt(sum((Yp-Y)*(Yp-Y))/Y.shape[0]))
           
 
 # methods for 1-dim examples like sin curve
    def initialise_line(self, lower,upper, size):        
        return(self.initialise(np.linspace(lower, upper, size).reshape(-1,1)))        


    def plotting(self, X, Y, color="b", confidence=False):          
        Yp , C       = self.predict(X);
        N = X.shape[0]
        #plt.scatter(X,Y,s=0.5,color='b', label='True Data')            
        ArgSort= X.reshape(1,-1)[0].argsort()
        
        plt.plot(X[ArgSort],Yp[ArgSort],color=color)        
        
        if(confidence):
            D=np.diag(C)
            if(np.min(D)<0):
                print("negative covariance:" + str(np.min(D)))
                C=np.sqrt(np.abs(D))
                C.reshape(-1,1)
                CI = 1.96 * C/np.sqrt(N)  
                CI = CI.reshape(-1,1)
                X=X.reshape((N,))
                Ylower = (Yp-CI).reshape((N,))
                Yupper = (Yp+CI).reshape((N,))
                plt.fill_between(X[ArgSort], Ylower[ArgSort], Yupper[ArgSort] , color='orange', alpha=.5)
        
            
    def training_function(self, n,f,lower,upper, plotting=False, color='r--'):
        x1          = (np.random.rand(n)*(upper-lower)+lower).reshape(-1,1)
        y1          = f(x1) + self.noise(x1)
        m1, C1      = self.recursiveGP(x1,y1)        
        Y , C       = self.predict(self.X);
        N=self.X.shape[0]
        if(plotting):
            D=np.diag(C)
            if(np.min(D)<0):
                print(np.min(D))
            C=np.sqrt(D)
            C.reshape(-1,1)
            CI = 1.96 * C/np.sqrt(N)  
            CI = CI.reshape(-1,1)
            X=self.X.reshape((N,))
            Ylower = (Y-CI).reshape((N,))
            Yupper = (Y+CI).reshape((N,))
            #plt.plot(X, m1, 'k')                    
            plt.plot(X,Y,color, label='Predicted Train Data')
            #plt.plot(X, Ylower, color='grey',alpha=.5)
            #plt.plot(X, Yupper, color='grey',alpha=.5)            
            plt.fill_between(X, Ylower, Yupper , color='orange', alpha=.5)
            plt.show()
            
        
        
    
        
    
    