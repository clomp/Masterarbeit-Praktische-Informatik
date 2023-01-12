# -*- coding: utf-8 -*-
"""
Implementation of an recursive GP regression proposed by M.F.Huber in

"Recursive Gaussian process: on-line regression and learning"
DOI 10.1016/j.patrec.2014.03.004

USAGE of class recursiveGP:
        instantiation:  sets kernel function and noise
                        variance     = sign variance of SE-kernel
                        lengthscale  = lengthscale  of SE-kernel
                        sigma        = Gaussian noise for y = g(x) + epsilon
                                        with epsilon ~ N(0,sigma**2)
        initialise:     fix a set of "basis" vectors X
        recursiveGP:    training of new datasets (xn,yn) will infere and update 
                        the mean and covariance of the basis vectors
        predict:        predicts mean and covariance of input xn
        

Created on Wed Nov 16 09:11:46 2022
@author: lomp
"""

import math
import itertools
import tqdm
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import gpflow
from gpflow.utilities import print_summary
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

class recursiveGP():    
    def __init__(self, variance=1.0 ,lengthscales=0.5, sigma = 0.01):
        """
        initilizing an object of the recursiveGP class according to Huber.
        Parameters
        ----------
        variance : float, optional            
        lengthscales : float or list of floats, optional            
        sigma : float, optional
            
        Parameters will be passed to the kernel        

        """        
        self.kernel         = gpflow.kernels.SquaredExponential(variance=variance,lengthscales=lengthscales)+gpflow.kernels.White(variance=sigma)
        self.covfunc        = self.kernel.K
        self.mean           = gpflow.mean_functions.Zero()
        self.sn             = sigma        
        self.noise          = lambda x : self.sn*np.random.randn()
            
    def initialise(self, X, options=None):
        """
        initialise provides a common interface for the GP testsuite and will 
        initialise the RGP object with a set of basis vectors and calculate
        their covariance matrix as well as its inverse
        
        Parameters
        ----------
        X : is an instance of the dataset class
            
        options : is a dictionary with two attributes:
                "num_base_vectors" - is the number of basis vectors
                "strategey" - can be "jb", "cl" or "kmeans"

        Returns
        -------
        None.

        """
        if(options is None):
            self.X = X
        else:
            self.X      = self.CreateBasisVectors(X, options["num_base_vectors"], options["strategy"])        
        
        self.mu     = self.mean(self.X).numpy()
        self.C      = self.covfunc(self.X, self.X).numpy()
        self.Kinv   = np.linalg.pinv(self.C)     

#------------------------------------------------------------------------------
    def CreateBasisVectors(self, dataobj, num_base_vectors=20, strategy="JB"):                
        
        if(strategy=="kmeans"):
            return(self._strategy_kmeans(dataobj, num_base_vectors))
        
        elif(strategy=="CL"):     
            return(self._strategy_cl(dataobj, num_base_vectors))
            
        elif(strategy=="JB"):
            return(self._strategy_jb(dataobj, num_base_vectors))
        
    def _strategy_kmeans(self, dataobj, num_base_vectors):
        kmeans = KMeans(n_clusters=num_base_vectors, random_state=42)    
        kmeans.fit(dataobj.XTrain)
        I=[[j for j in range(dataobj.XTrain.shape[0]) if kmeans.labels_[j]==i] for i in range(num_base_vectors)]
        Centers=[[np.mean(dataobj.XTrain[I[i]][:,j]) for j in range(dataobj.input_dim)] for i in range(num_base_vectors)]
        return(np.array(Centers))    

    def _strategy_cl(self, dataobj, num_base_vectors):
        lb,ub = dataobj.get_input_bounds()    
        num_sep = int(math.pow(num_base_vectors,1/dataobj.input_dim)) 
        list_base_coordinates=[ list(np.linspace(lb[i], ub[i], num_sep))  for i in range(dataobj.input_dim)]
        list_base_vectors=[]
        for b in itertools.product(*list_base_coordinates):
            list_base_vectors.append(b)    
        return(np.array(list_base_vectors))
    
    def _strategy_jb(self, dataobj, num_base_vectors):
        lb,ub = dataobj.get_input_bounds()
        list_base_coordinates=[ np.linspace(lb[i], ub[i], num_base_vectors)  for i in range(dataobj.input_dim)]
        for ls in list_base_coordinates:
            np.random.shuffle(ls)
        return(np.vstack(tuple(list_base_coordinates)).T)
    
#------------------------------------------------------------------------------

    def train_batch(self, xt, yt, batch):            
        for i in tqdm.tqdm(range(0, xt.shape[0]+1, batch)):  
            self.recursiveGP(xt[i:i+batch,:], yt[i:i+batch])

    def recursiveGP(self, xn, yn):
        Knx         = self.covfunc(xn,self.X).numpy()
        Kn          = self.covfunc(xn,xn).numpy()
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
        G = Cxn @ np.linalg.pinv(Cn + self.sn * np.eye(yn.shape[0]))  
        # Update mean and covariance of basis vectors
        self.mu     = self.mu  + G@(yn - mn)
        self.C      = self.C - G@Cxn.T    
        return(self.mu,self.C)
    
    
    def predict(self, xn):
        Knx         = self.covfunc(xn,self.X).numpy()
        Kn          = self.covfunc(xn,xn).numpy()
        mn          = self.mean(xn)
        mx          = self.mean(self.X)        
        J           = Knx @ self.Kinv  
        D           = Kn - J @ Knx.T
        # covariance of new data
        Cxn         = self.C @  J.T
        Cn          = D + J @ Cxn;
        # mean and covariance of new data
        mu_pred     = mn + J @ (self.mu - mx)        
        C_pred      = Cn + self.sn*np.eye(Cn.shape[0]);
        return(mu_pred,C_pred)
 
        
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
            
        
        
    
        
    
    