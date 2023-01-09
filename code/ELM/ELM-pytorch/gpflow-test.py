# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:39:49 2022

@author: lomp
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from gpflow.utilities import print_summary

from sklearn.model_selection import train_test_split

number=500

X = np.linspace(-1,2,3*number).reshape(-1,1)
Y = np.sin(4*np.pi*X).reshape(-1,1)

X_links = X[:number]
Y_links = Y[:number]

X_mitte = X[number:2*number]
Y_mitte = Y[number:2*number]

X_rechts = X[2*number:]
Y_rechts = Y[2*number:]

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_mitte,Y_mitte, test_size=0.2, random_state=42)


#build model
#k = gpflow.kernels.Matern52()
k = gpflow.kernels.SquaredExponential(variance=1.0,lengthscales=1.0)
m = gpflow.models.GPR(data=(X_Train, Y_Train), kernel=k, mean_function=None)
m.likelihood.variance.assign(1E-05)

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(
    m.training_loss, m.trainable_variables, options=dict(maxiter=50)
)
print_summary(m)

    #Predict
Y_Train_mean, Y_Train_var = m.predict_f(X_Train)
Y_Test_mean, Y_Test_var = m.predict_f(X_Test)

Y_links_mean, Y_links_var = m.predict_f(X_links)
Y_rechts_mean, Y_rechts_var = m.predict_f(X_rechts)

    #Plot

plt.figure()

plt.plot(X,Y,'y',label='True Data')

plt.plot(X_Train, Y_Train_mean,'b.', label='Predicted Train Data')
plt.fill_between(
    X_Train[:, 0],
    Y_Train_mean[:, 0] - 1.96 * np.sqrt(Y_Train_var[:, 0]),
    Y_Train_mean[:, 0] + 1.96 * np.sqrt(Y_Train_var[:, 0]),
    color="lightblue",
    alpha=0.2,
)

plt.plot(X_Test, Y_Test_mean,'r.', label='Predicted Test Data')
plt.fill_between(
    X_Test[:, 0],
    Y_Test_mean[:, 0] - 1.96 * np.sqrt(Y_Test_var[:, 0]),
    Y_Test_mean[:, 0] + 1.96 * np.sqrt(Y_Test_var[:, 0]),
    color="lightgreen",
    alpha=0.2,
)

plt.plot(X_links, Y_links_mean, 'g', label='Extrapolated Predicted Data')
plt.fill_between(
    X_links[:, 0],
    Y_links_mean[:, 0] - 1.96 * np.sqrt(Y_links_var[:, 0]),
    Y_links_mean[:, 0] + 1.96 * np.sqrt(Y_links_var[:, 0]),
    color=(0.5,0.5,0.5),
    alpha=0.5,
)

plt.plot(X_rechts, Y_rechts_mean, 'g', label='Extrapolated Predicted Data')
plt.fill_between(
    X_rechts[:, 0],
    Y_rechts_mean[:, 0] - 1.96 * np.sqrt(Y_rechts_var[:, 0]),
    Y_rechts_mean[:, 0] + 1.96 * np.sqrt(Y_rechts_var[:, 0]),
    color=(0.5,0.5,0.5),
    alpha=0.5,
)
        
plt.legend()
plt.title('GPR Model')
plt.show()

## generate test points for prediction
#xx = np.linspace(-1, 2, 1000).reshape(1000, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points

# mean, var = m.predict_f(xx)

## generate 10 samples from posterior
#tf.random.set_seed(1)  # for reproducibility
#samples = m.predict_f_samples(xx, 100)  # shape (10, 100, 1)

## plot
#plt.figure(figsize=(12, 6))
#plt.plot(X, Y, "r", mew=1)
#plt.plot(xx, mean, "C0", lw=1)
#plt.fill_between(
#    xx[:, 0],
#    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
#    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
#    color="C1",
#    alpha=0.2,
#)

#plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.01)
#_ = plt.xlim(-1, 2)

