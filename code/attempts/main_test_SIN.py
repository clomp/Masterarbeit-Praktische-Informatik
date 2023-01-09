# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#GPflow is newer implementation of Gauss process regression and based on GPy
#See: https://gpflow.readthedocs.io/en/master/index.html for documentation
#Further there is GPyTorch, an implementation of GPR based on PyTorch
#See: https://docs.gpytorch.ai/en/stable/index.html for documentation
#Other literatur
#https://github.com/Bigpig4396/Incremental-Gaussian-Process-Regression-IGPR
#https://mediatum.ub.tum.de/doc/1364689/1364689.pdf
#https://www.mathworks.com/matlabcentral/fileexchange/46748-kernel-methods-toolbox
#https://www.mathworks.com/help/stats/incremental-learning-overview.html
import numpy as np
#import matplotlib
from matplotlib import pyplot as plt
import GPy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import layers
from tensorflow import keras


def predict_Confidence(Y):
    Y_lower = Y[0] - 1.96 * Y[1] / np.sqrt(500)
    Y_upper = Y[0] + 1.96 * Y[1] / np.sqrt(500)
    Y_Conf = np.hstack((Y_lower,Y_upper))
    return Y_Conf 


GPR = True
ANN = False
Extrapolation = True

#------------------------------------------------------------------------------
#Read data into numpy array
X = np.linspace(0,1,500)
X = np.array([X]).T
f = 4*np.pi
Y = np.sin(X*f)
#------------------------------------------------------------------------------
#Extrapolation Data only used for Testing
X_ext = np.hstack((np.linspace(-1,0,500), np.linspace(1,2,500)))
X_ext = np.array([X_ext]).T
Y_ext = np.sin(X_ext*f)
#------------------------------------------------------------------------------
#Split Data into Training and Testing
#Train-Test Split with ratio test_size, 
#random_state should be set so that the shuffling is reproducible, e.g. =42
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, test_size=0.2, random_state=42)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# GAUSS MODEL
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if GPR:
    #Define Kernel Function
    kernel1 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    kernel2 = GPy.kern.White(input_dim=1, variance=1.)
    kernel3 = GPy.kern.Bias(input_dim=1, variance=1.)
    # kernel = kernel1 + kernel2 + kernel3
    kernel = kernel1
    
    #Build Model
    my_model = GPy.models.GPRegression(X_Train,Y_Train,kernel=kernel)
    
    #Train Model
    my_model.optimize_restarts(num_restarts=5)
    my_model.optimize
    
    #Predict
    Y_Train_pred = my_model.predict(X_Train)
    Y_Test_pred  = my_model.predict(X_Test)
    
    #95% Confidence Interval
    Y_Train_pred_Conf = predict_Confidence(Y_Train_pred)
    Y_Test_pred_Conf = predict_Confidence(Y_Test_pred)
       
    #Plot
    plt.figure()
    plt.plot(X,Y,'k.-',label='True Data')
    plt.plot(X_Train,Y_Train_pred[0],'b.', label='Predicted Train Data')
    plt.plot(X_Test,Y_Test_pred[0],'r.', label='Predicted Test Data')
    
    plt.plot(X_Train, Y_Train_pred_Conf[:,0],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #lower bound confidence
    plt.plot(X_Train, Y_Train_pred_Conf[:,1],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #upper bound confidence
    plt.plot(X_Test, Y_Test_pred_Conf[:,0],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #lower bound confidence
    plt.plot(X_Test, Y_Test_pred_Conf[:,1],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #upper bound confidence
    
    #Test Extrapolation Capability
    if Extrapolation:
        Y_ext_pred = my_model.predict(X_ext)
        Y_ext_pred_Conf = predict_Confidence(Y_ext_pred)
        plt.plot(X_ext,Y_ext, 'k.', label='Extrapolated True Data')
        plt.plot(X_ext, Y_ext_pred[0], 'g.', label='Extrapolated Predicted Data')
        plt.plot(X_ext, Y_ext_pred_Conf[:,0],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #lower bound confidence
        plt.plot(X_ext, Y_ext_pred_Conf[:,1],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #upper bound confidenc
        
    plt.legend()
    plt.title('GPR Model')
    plt.show()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#ANN MODEL
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if ANN:
    #Build model
    inputs = keras.Input(shape=(1,))
    model = keras.Sequential()
    model.add(layers.Dense(100, input_dim=1, activation='relu'))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(1, activation=None))
    
    #Train model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_Train,Y_Train, validation_data=(X_Test,Y_Test), epochs=1000, batch_size=50)
    
    #Predict
    Y_Train_pred_ANN = model.predict(X_Train)
    Y_Test_pred_ANN = model.predict(X_Test)
    
    #Plot
    plt.figure()
    plt.plot(X,Y,'k.-',label='True Data')
    plt.plot(X_Train,Y_Train_pred_ANN,'b.', label='Predicted Train Data')
    plt.plot(X_Test,Y_Test_pred_ANN,'r.', label='Predicted Test Data')
    
    #Test Extrapolation Capability
    if Extrapolation:
        Y_ext_pred_ANN = model.predict(X_ext)
        plt.plot(X_ext,Y_ext, 'k.', label='Extrapolated True Data')
        plt.plot(X_ext, Y_ext_pred_ANN, 'g.', label='Extrapolated Predicted Data')
    
    plt.legend()
    plt.title('ANN Model')
    plt.show()

