# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:13:26 2023

the dataset class provides
    - loading a dataset from a csv file and split it into train/test pairs
    - saving train/test pairs into separate files for later use
    - normalizing and denormalizing data

@author: lomp
"""

import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import itertools
import math

DATASET_META    = [(0.0), (6,2000),(12,2000), (12,2000), (6,None), (6,None), (9,None), (9,None)]

class dataset():
    def __init__(self, data_id, 
                     DATASET_PATH    = "../datasets/",
                     DATASET         = "dataset",
                     OUTPUT_PATH     = "output/",
                     META_PATH       = "meta/",                 
                 ):
        self.id=data_id
        self.filename        = DATASET+str(data_id)
        self.data_filename   = DATASET_PATH+self.filename+".csv"
        self.meta_filename   = META_PATH + self.filename+"_meta.json"
        self.train_filename  = META_PATH + self.filename+"_train.csv"
        self.test_filename   = META_PATH + self.filename+"_test.csv"        
        self.input_dim = DATASET_META[data_id][0]
        self.limitations = DATASET_META[data_id][1]
    
    def normalize(self, A,length,offset):
        return((A-offset)/length)

    def denormalize(self, A,length,offset):
        return((A*length)+offset)
    
    def load_dataframe(self, reuse=False, test_size=0.2, save=True, default_lengthscale=0.5, default_variance=1.0):
        if(os.path.isfile(self.meta_filename) and reuse):        
            TestData  = np.loadtxt(self.test_filename,delimiter=",")
            self.XTest, self.YTest = np.split(TestData, [self.input_dim],axis=1)
                
            TrainData = np.loadtxt(self.train_filename,delimiter=",")
            self.XTrain, self.YTrain  = np.split(TrainData, [self.input_dim], axis=1)
            
            f=open(self.meta_filename)
            self.ds_offset, self.ds_length, self.Lengthscales, self.Variances = json.load(f)
            f.close()
        else:
            dataframe = np.loadtxt(self.data_filename,delimiter=",")
            dimension = dataframe.shape[1]    
            self.ds_offset=[np.mean(dataframe[:,i:i+1]) for i in range(dimension)]
            self.ds_length=[np.std(dataframe[:,i:i+1]) for i in range(dimension)]      
            
            dataframe = self.normalize(dataframe, self.ds_offset, self.ds_length)            
            X_values, Y_values = np.split(dataframe, [self.input_dim], axis=1)
            
            self.XTrain, self.XTest , self.YTrain, self.YTest = train_test_split(X_values,Y_values, test_size=test_size, random_state=42)      
            
            self.Lengthscales=[default_lengthscale]*self.input_size
            self.Variances=[default_variance]*self.input_size
            if(save):
                np.savetxt(self.test_filename, np.hstack((self.XTest, self.YTest)), delimiter=",")
                np.savetxt(self.train_filename, np.hstack((self.XTrain, self.YTrain)), delimiter=",")
                out_file = open(self.meta_filename, "w")
                json.dump([self.ds_offset,self.ds_length,self.Lengthscales,self.Variances], out_file)
                out_file.close()   
        
        self.lb=[np.min(self.XTrain[:,i]) for i in range(self.input_dim)]
        self.ub=[np.max(self.XTrain[:,i]) for i in range(self.input_dim)]



