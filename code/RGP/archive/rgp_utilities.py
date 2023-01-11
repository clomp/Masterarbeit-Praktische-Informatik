# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:48:34 2022

@author: lomp
"""
import numpy as np
import json, os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import itertools
import math


# utilities
#------------------------------------------------------------------------------
TRAIN_DATA = "train_data.csv"
TEST_DATA  = "test_data.csv"


def myprint(SILENT, txt):
    if(SILENT==False):
        print(txt)

def PrepareData(dataframe,input_size, output_size, index_sep, test_size_proportion=0.2, load=False, silentmode=False):

    if(load and os.path.isfile(TEST_DATA)):
        myprint(silentmode,"read train and test data from file.")
        TestData = np.loadtxt(TEST_DATA,delimiter=",")
        TrainData = np.loadtxt(TRAIN_DATA,delimiter=",") 
        XTest  = TestData[ : , : input_size]
        YTest  = TestData[ : , index_sep : index_sep + output_size]    
        XTrain = TrainData[ : , : input_size]
        YTrain = TrainData[ : , index_sep : index_sep + output_size]    
    else:
        myprint(silentmode,"... splitting of train- and testdata.")            

        # normalize data to obtain zero mean and std 1
        num_data = dataframe.shape[1]
        ds_offset=[np.mean(dataframe[:,i:i+1]) for i in range(num_data)]
        ds_length=[np.std(dataframe[:,i:i+1]) for i in range(num_data)]  
        for i in range(num_data):
            if(ds_length[i]!=0):
                dataframe[:,i:i+1] = (dataframe[:,i:i+1]-ds_offset[i])/ds_length[i]        
        # split data in input and output
        X_values    = dataframe[ : , 0 : input_size]
        Y_values    = dataframe[ : , index_sep : index_sep + output_size]
    
        # splitt data into train and test ets
        XTrain, XTest , YTrain, YTest = train_test_split(X_values,Y_values, test_size=test_size_proportion, random_state=42)
        # write train and test data to file
        np.savetxt(TEST_DATA,np.hstack((XTest,YTest)),delimiter=",")
        np.savetxt(TRAIN_DATA,np.hstack((XTrain,YTrain)),delimiter=",")    
    
    return(XTrain, XTest, YTrain, YTest)
 
#------------------------------------------------------------------------------   
def plotting(evaluation, label, analysis, nbins=100,xlims=5):
    output_size=len(evaluation)
    if (output_size>1):
        figure, ax = plt.subplots(1,output_size, figsize = (10,10))    
    else:
        figure, ax = plt.subplots(1,2, figsize = (10,10))            
    figure.suptitle(label+"Total MSE = "+str(analysis["totalMSE"]))    
    for i in range(output_size):
        ax[i].hist(evaluation[i],bins=nbins)    
        ax[i].set_xlim([0, xlims])  
        ax[i].set_title("MSE="+str(analysis[i][0])+"\n STD="+str(analysis[i][1])+"\n max="+str(analysis[i][2]))
    plt.savefig(label+".pdf", format="pdf", bbox_inches="tight")
    plt.show()    
    
#------------------------------------------------------------------------------
# calculate absolut value of difference between predicted and true data
# scales with scalwer=1000 (from meters to mm)
def getError(predict_fns, XTest, YTest, scalar=1000):
    Ntest = XTest.shape[0]
    output_size=len(predict_fns)
    evaluation = np.empty((output_size,Ntest))
    for i in range(output_size): 
        Mean,_  = predict_fns[i](XTest) 
        Y_pred = Mean.reshape((Ntest,))                                     
        evaluation[i] = Y_pred - YTest[i]
    return(np.abs(scalar*evaluation).T)

#------------------------------------------------------------------------------
def analyzeError(diff,silentmode):
    output_size=diff.shape[1]
    analysis={'totalMSE':MSE(diff)}    
    myprint(silentmode,"\n Mean Squared Error:" + str(analysis['totalMSE'])+"\n i \t MSE \t STD \t max")    
    for i in range(output_size):
        analysis[i]=(MSE(diff[:,i]), np.std(diff[:,i]),np.max(diff[:,i])) 
        myprint(silentmode,str(i)+"\t"+str(analysis[i][0])+"\t"+str(analysis[i][1])+"\t"+str(analysis[i][2]))
    return(analysis)
                    
#------------------------------------------------------------------------------
def MSE(D):
    return(np.mean([np.linalg.norm(x)**2 for x in D]))

#------------------------------------------------------------------------------
def NLL(M,C,Y):
    D=np.diag(C)
    s= sum(np.log(D[i]) + 1/D[i] * (Y[i]-M[i])**2  for i in range(len(D)))
    return(0.5*s)

#------------------------------------------------------------------------------
def CreateBasisVectors(XTrain, input_size, version="JB", num_base_vectors=20):        
    N=len(XTrain)
    lb=[np.min(XTrain[:,i]) for i in range(input_size)]
    ub=[np.max(XTrain[:,i]) for i in range(input_size)]    
    if(version=="kmeans"):
        kmeans = KMeans(n_clusters=num_base_vectors, random_state=42)    
        kmeans.fit(XTrain)
        I=[[j for j in range(N) if kmeans.labels_[j]==i] for i in range(num_base_vectors)]
        Cent=[[np.mean(XTrain[I[i]][:,j]) for j in range(input_size)] for i in range(num_base_vectors)]
        return(np.array(Cent))
    elif(version=="CEL"):     
        num_sep = int(math.pow(num_base_vectors,1/input_size)) # take the nth root of num_base_vectors
        list_base_coordinates=[ list(np.linspace(lb[i], ub[i], num_sep))  for i in range(input_size)]
        list_base_vectors=[]
        for b in itertools.product(*list_base_coordinates):
            list_base_vectors.append(b)    
        return(np.array(list_base_vectors))
    elif(version=="JB"):
        list_base_coordinates=[ np.linspace(lb[i], ub[i], num_base_vectors)  for i in range(input_size)]
        for ls in list_base_coordinates:
            np.random.shuffle(ls)
        return(np.vstack(tuple(list_base_coordinates)).T)
