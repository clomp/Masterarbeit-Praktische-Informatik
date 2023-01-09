# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:57:27 2022

@author: lomp
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import matplotlib.pyplot as plt
import numpy as np
from recursiveGP import recursiveGP
from sklearn.model_selection import train_test_split
import tqdm

f = lambda x: np.sin(2*x)  
#f = lambda x: x/2 + (25*x/(1+x**2))*np.cos(x)



N=20; lower=0; upper=6
#N=20; lower=-10; upper=10

X_values = np.linspace(lower,upper, 500).reshape(-1,1);
Y_values = f(X_values)

XTrain, XTest , YTrain, YTest = train_test_split(X_values,
                                                 Y_values, 
                                                 test_size=0.2, 
                                                 random_state=42)

batch = 20
Ntraining = 100

modelRGP = recursiveGP(variance=1.0, lengthscales=0.06, sigma=0.1)

#modelRGP.initialise_line(lower, upper, N)
print("Initialise RGP")
xg = np.linspace(lower,upper,N).reshape(-1,1)

modelRGP.initialise(xg)
print("done")

print("Initialise full GP")
modelGP = modelRGP.estimate_hyper(xg,f(xg))
print("done")

YTest_pred, _ = modelRGP.predict(xg)

print("Loss before recursiveGP:" + str(modelRGP.lossRMSE(f(xg), YTest_pred))+"\n")
print("Training:")
num_dataset = XTrain.shape[0]
for i in tqdm.trange(Ntraining):
    #I = np.random.choice(num_dataset, batch, replace=False)  
    #modelRGP.recursiveGP(XTrain[I], YTrain[I])         
    Xgen=(np.random.randn(batch)*(upper-lower)).reshape(-1,1)
    modelRGP.recursiveGP(Xgen,f(Xgen))
        
plt.plot(X_values, Y_values, color="grey");
modelRGP.plotting(xg,f(xg),confidence=True)

YTest_pred, _ = modelRGP.predict(xg)
print("\n Loss after recursiveGP:" + str(modelRGP.lossRMSE(xg, YTest_pred)))

YTest_GP,_=modelGP.predict_f(xg)
print("\n Loss of full GP:" + str(modelRGP.lossRMSE(f(xg), YTest_GP)))