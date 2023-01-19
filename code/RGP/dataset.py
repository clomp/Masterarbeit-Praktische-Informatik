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

DATASET_META    = [(0.0,0), (6,6,2000),(9,12,2000), (9,12,2000), (6,6,None), (6,6,None), (9,9,None), (9,9,None)]

class dataset():
    def __init__(self, data_id, 
                     gprname="",
                     DATASET_PATH    = "../../../datasets/",
                     DATASET         = "dataset",
                     OUTPUT_PATH     = "output/",
                     META_PATH       = "meta/"
                 ):
        self.id=data_id
        self.filename        = DATASET+str(data_id)
        self.data_filename   = DATASET_PATH+self.filename+".csv"
        self.meta_filename   = META_PATH + self.filename+"_meta.json"
        self.train_filename  = META_PATH + self.filename+"_train.csv"
        self.test_filename   = META_PATH + self.filename+"_test.csv"     
        self.output_filename = OUTPUT_PATH + self.filename+"_"+gprname+".pdf"
        self.input_dim       = DATASET_META[data_id][0]
        self.output_offset   = DATASET_META[data_id][1]
        self.limitations     = DATASET_META[data_id][2]
        self.output_path     = OUTPUT_PATH
    
    def _normalize(self, A,offset,length):
        if(length!=0): 
            try:
                return((A-offset)/length)
            except:
                print(A,offset,length)
                return(A)
        else:
            return(A)

    def _denormalize(self, A,offset,length):
        return((A*length)+offset)
    
    def load_dataframe(self, reuse=False, test_size=0.2, save=False, default_lengthscale=0.5, default_variance=1.0):
        if(os.path.isfile(self.meta_filename) and reuse):  
            print("Pre-calcuated Test/Train pairs will be loaded.")
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
            
            dataframe = self._normalize(dataframe, self.ds_offset, self.ds_length)            
            #X_values, Y_values = np.split(dataframe, [self.input_dim], axis=1)
            X_values = dataframe[ : , : self.input_dim]
            Y_values = dataframe[ : , self.output_offset: ]
            
            self.XTrain, self.XTest , self.YTrain, self.YTest = train_test_split(X_values,Y_values, test_size=test_size, random_state=42)      
            
            self.Lengthscales=[[default_lengthscale]*self.input_dim]*(dimension-self.output_offset)
            self.Variances=[default_variance]*(dimension-self.output_offset)
            if(save):
                np.savetxt(self.test_filename, np.hstack((self.XTest, self.YTest)), delimiter=",")
                np.savetxt(self.train_filename, np.hstack((self.XTrain, self.YTrain)), delimiter=",")
                out_file = open(self.meta_filename, "w")
                json.dump([self.ds_offset,self.ds_length,self.Lengthscales,self.Variances], out_file)
                out_file.close()
        # denormalizing the y-values of the test data for calculating the error
        self.YTest_denormalized = self._denormalize(self.YTest, self.ds_offset[self.output_offset:], self.ds_length[self.output_offset:]) 
    
    def get_input_bounds(self):
        lb=[np.min(self.XTrain[:,i]) for i in range(self.input_dim)]
        ub=[np.max(self.XTrain[:,i]) for i in range(self.input_dim)]
        return(lb,ub)    
    
    def get_difference(self, Ypred, coordinate):
        length      = self.ds_length[coordinate+self.output_offset]
        offset      = self.ds_offset[coordinate+self.output_offset]    
#        YTst        = self._denormalize(self.YTest[:,coordinate:coordinate+1], offset,  length)
        YTst       = self.YTest_denormalized[:,coordinate] 
        YTst_pred   = self._denormalize(Ypred, offset, length)
        return(np.abs(YTst-YTst_pred))  

    def analyze_error(self, predictions, scaling):
        num_models=len(predictions)
        N=predictions[0].shape[0]                
        difference  = [scaling * self.get_difference(predictions[i],i).reshape((N,)) for i in range(num_models)]
        difference = np.array(difference).T
        totalMSE = np.square(np.linalg.norm(difference, axis=1)).mean()
        componentwiseErrors = np.array([
                        np.square(difference).mean(axis=0), 
                        np.std(difference,axis=0), 
                        np.max(difference,axis=0)]
                    ).T
        return(difference, totalMSE, componentwiseErrors)    
    
    def print_analysis(self,difference, totalMSE, componentwiseErrors, savePDF=True, nbins=100):        
        print("Total MSE:" + str(totalMSE))
        print("MSE, STD and MAX")
        for i in range(len(componentwiseErrors)):
            print(i, componentwiseErrors[i])
        output_size=difference.shape[0]
        xlims=max(componentwiseErrors[:,2])
        if (output_size>1):
            figure, ax = plt.subplots(1,output_size, figsize = (10,10))    
        else:
            figure, ax = plt.subplots(1,2, figsize = (10,10))            
        figure.suptitle(self.filename+": Total MSE = "+str(totalMSE))    
        for i in range(output_size):
            ax[i].hist(difference[i],bins=nbins)    
            ax[i].set_xlim([0, xlims])  
            ax[i].set_title(  "MSE=" + str(componentwiseErrors[i,0])
                            + "\n STD=" + str(componentwiseErrors[i,1]) 
                            + "\n max=" + str(componentwiseErrors[i,2])
                            )
        if(savePDF):
            plt.savefig(self.output_filename, format="pdf", bbox_inches="tight")
        plt.show()  

    def CreateBasisVectors(self, num_base_vectors=20, strategy="JB"):                
        
        if(strategy=="kmeans"):
            return(self._strategy_kmeans(num_base_vectors))
        
        elif(strategy=="CL"):     
            return(self._strategy_cl(num_base_vectors))
            
        elif(strategy=="JB"):
            return(self._strategy_jb(num_base_vectors))
        
    def _strategy_kmeans(self, num_base_vectors):
        kmeans = KMeans(n_clusters=num_base_vectors, random_state=42)    
        kmeans.fit(self.XTrain)
        I=[[j for j in range(self.XTrain.shape[0]) if kmeans.labels_[j]==i] for i in range(num_base_vectors)]
        Centers=[[np.mean(self.XTrain[I[i]][:,j]) for j in range(self.input_dim)] for i in range(num_base_vectors)]
        return(np.array(Centers))    

    def _strategy_cl(self,  num_base_vectors):
        lb,ub = self.get_input_bounds()    
        num_sep = int(math.pow(num_base_vectors,1/self.input_dim)) 
        list_base_coordinates=[ list(np.linspace(lb[i], ub[i], num_sep))  for i in range(self.input_dim)]
        list_base_vectors=[]
        for b in itertools.product(*list_base_coordinates):
            list_base_vectors.append(b)    
        return(np.array(list_base_vectors))
    
    def _strategy_jb(self, num_base_vectors):
        lb,ub = self.get_input_bounds()
        list_base_coordinates=[ np.linspace(lb[i], ub[i], num_base_vectors)  for i in range(self.input_dim)]
        for ls in list_base_coordinates:
            np.random.shuffle(ls)
        return(np.vstack(tuple(list_base_coordinates)).T)
    
        
    
        

