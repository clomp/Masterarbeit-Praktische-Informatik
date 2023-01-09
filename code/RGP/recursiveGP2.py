# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:03:47 2022
implementing my own covfunc and mean functions
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
        self.lengthscales   = lengthscales
        self.variance       = variance
        self.sn             = sigma
        self.kernel         = gpflow.kernels.SquaredExponential(variance=variance,lengthscales=lengthscales)
        #self.kernel         = gpflow.kernels.Matern12(variance=1.0,lengthscales=lengthscales)        
        self.covfunc        = self.covfunc2 #self.kernel.K        
        self.mean           = lambda X: np.apply_along_axis(lambda x: 0, 1,X).reshape((X.shape[0],1))
        self.noise          = lambda x : self.sn*np.random.randn()

    def covfunc_vector(self, x1, x2):
        return(self.variance * np.exp(-np.linalg.norm(x1-x2)**2/(2*self.lengthscales**2)))

    def covfunc2(self, X1,X2):
        n=X1.shape[0]
        m=X2.shape[0]
        Z=np.zeros((n,m))
        for i in range(n):
            Z[i]=np.apply_along_axis(lambda x2: self.covfunc_vector(X1[i],x2),1,X2)            
        return(Z)
            
    def initialise(self, X):
        self.X      = X
        self.mu     = self.mean(self.X)
        self.C      = self.covfunc(self.X, self.X) #.numpy()
        self.Kinv   = np.linalg.pinv(self.C)     
    
    def recursiveGP(self, xn, yn):
        Knx         = self.covfunc(xn,self.X) #.numpy()
        Kn          = self.covfunc(xn,xn) #.numpy()
        mn          = self.mean(xn) 
        mx          = self.mean(self.X)        
        J           = Knx @ self.Kinv  
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
        Knx         = self.covfunc(xn,self.X) #.numpy()
        Kn          = self.covfunc(xn,xn) #.numpy()
        mn          = self.mean(xn)    
        mx          = self.mean(self.X)
        J           = Knx @ self.Kinv  
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
            
        
        
    
        
    
    