# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:32:49 2022

@author: lomp
"""


from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d 
import gpflow
from recursiveGP import recursiveGP
import tqdm
from gpflow.utilities import print_summary

def my_plot3d(x,y,z,ypred=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, color="blue", edgecolor='none') #cmap='viridis',
    if(ypred is not None):
        ax.plot_surface(x,y,ypred, color="red", edgecolor='none') #cmap='viridis',
    plt.show()


def my_wiredplot(wf,x,y,z,color):   
    wf.plot_wireframe(x,y,z, rstride=2, cstride=2, color=color) 

f = lambda u,v : u**2-v**2

upper = -4
lower = 4
N=20

I=np.linspace(lower, upper, N)

xg = np.array([ [a,b] for a in I for b in I])

x,y = np.meshgrid(I,I)
z = f(x,y)


modelRGP = recursiveGP(variance=1., lengthscales=[0.4,0.3], sigma=0.1)
print("Initialise RGP")

modelRGP.initialise(xg)
print("done")


modelGP = gpflow.models.GPR(data=(xg, z.flatten('F').reshape((N*N,1))), 
                         kernel= gpflow.kernels.SquaredExponential(variance=1.0,lengthscales=[0.5,0.5]),
                         mean_function=gpflow.mean_functions.Zero())      
modelGP.likelihood.variance.assign(0.1)        
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(modelGP.training_loss, modelGP.trainable_variables)        
print_summary(modelGP)              


Y_pred, _ = modelRGP.predict(xg)
Y = np.array([ f(a,b) for [a,b] in xg]).reshape((N*N,1))


print("Loss before recursiveGP:" + str(modelRGP.lossRMSE(Y, Y_pred))+"\n")

batch=20
Ntraining=100

print("Training:")
for i in tqdm.trange(Ntraining):
    a=(np.random.randn(batch)*(upper-lower))
    b=(np.random.randn(batch)*(upper-lower))
    xa,yb = np.meshgrid(a,b)
    z = f(xa,yb)    
    xgen = np.array([ [u,v] for u in a for v in b  ])
    modelRGP.recursiveGP(xgen,z.flatten('F').reshape((batch*batch,1)))
    

Y_true = np.array([ f(a,b) for [a,b] in xg]).reshape((N*N,1))
Y_pred, _ = modelRGP.predict(xg)
Y_full,_ = modelGP.predict_f(xg)


print("\n Loss after recursiveGP:" + str(modelRGP.lossRMSE(Y, Y_pred))+"\n")
print("\n Loss of  full GP:" + str(modelRGP.lossRMSE(Y, Y_full))+"\n")


x,y = np.meshgrid(I,I)
fig = plt.figure() 
wf = fig.add_subplot(111, projection='3d')     
my_wiredplot(wf,x,y,(Y_true.reshape((N,N))),color="grey")
my_wiredplot(wf,x,y,(Y_pred.reshape((N,N))).numpy(), color='red') 
my_wiredplot(wf,x,y,(Y_full.reshape((N,N))).numpy(), color='blue') 
plt.show() 


        
# plt.plot(xg, f(xg), color="grey");
# modelRGP.plotting(xg,f(xg),color="r", confidence=True)

# YTest_pred, _ = modelRGP.predict(xg)
# print("\n Loss after recursiveGP:" + str(modelRGP.lossRMSE(xg, YTest_pred)))
