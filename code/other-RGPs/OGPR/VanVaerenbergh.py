#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 21:53:10 2022

@author: christianlomp
"""

import numpy as np
import matplotlib.pyplot as plt
import GPy


# model M_t = (D_t, mu_t, Sigma_t, Q_t)
# D_t = (x_1, ... x,_t)
# mu_i and Sigma_t mean and covariance of posterior p(f_t | D_t) = N(mu_t, Sigma_t)
# Q_t = K_t^{-1} inverse of covariance matrix

# sigma_n - variance of Gauss noise
# f_t latent function at time t

# variance of white noise
sigma = 0.01



# latent function
def f(x):
    return(np.sin(2*np.pi*x))

def f_observed(x):
    return(f(x) + sigma*np.random.randn())
  
Interval = np.linspace(-1,1,5000)

Xstart = np.random.choice(Interval, 10)
Ystart = f_observed(Xstart)

 #Define Kernel Function
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
 
 #Build Model
model = GPy.models.GPRegression(np.array([Xstart]).T,np.array([Ystart]).T,kernel=kernel)
 
 #Train Model
model.optimize_restarts(num_restarts=5)
model.optimize
 
print(model)

sigma= model.Gaussian_noise.variance
lengthscale=model.rbf.lengthscale
variance=model.rbf.variance


def my_kernel(a,b):
    return(variance*np.exp(-(a-b)**2/(2*lengthscale)))
    

def kern(A,B):
    mkern=model.kern.K
    return(mkern(np.array([A]).T,np.array([B]).T))
        
mu = model.rbf

X_online = np.random.choice(Interval, 100)
Y_online = f_observed(X_online)



def get_q_t(Q_t,k_t):
    return(Q_t@k_t)

def initialise(x,y):
    kxx     = kern(np.array([x]),np.array([x]))
    mu      = y*kxx/(sigma + kxx)
    Sigma   = np.array(kxx - kxx*kxx/(sigma + kxx))
    Q       = np.array(1/kxx)
    return(mu, Sigma, Q)

mu, Sigma, Q = initialise(X_online[0], Y_online[0])

def get_gamma_t(k_t, q_t):
    return(k_t - k_t.T @ q_t )

def update_Q(Q_t, gamma_t, q_t):
    Qh = np.vstack((np.hstack((Q_t,[0])),np.array([0,0])))  
    return(Qh + (1/gamma_t)*np.vstack((q_t,[-1]))@np.vstack((q_t,[-1])).T)

def update(D, mu, Sigma, Q,x,y):
    k_t = kern(D, x)
    q_t = Q @ k_t
    y_t = k_t.T @ Q @ mu
    return(k_t,q_t,y_t)


    



plt.plot(Interval, f(Interval))
plt.scatter(X_online,Y_online,color="g")
plt.show()