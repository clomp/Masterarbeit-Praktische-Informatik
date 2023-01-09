#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from math import sqrt, ceil
import csv

class ELM_model:
    
    def __init__(self,data,labels,hidden_size,activation="relu"):
        self.data = data
        self.labels = labels
        self.hidden_size=hidden_size
        if(activation=="relu"):
            self.activation=self.relu
        elif(activation=="sigmoid"):
            self.activation=self.sigmoid
        else:
            print("Error: no such activation rule implemented")
            return()
        
        #create model
        _, input_size = self.data.shape
        
#        self.input_weights = np.random.normal(size=[input_size,hidden_size])
#        self.biases = np.random.normal(size=[hidden_size])

        self.input_weights = np.random.rand(input_size,hidden_size)*2-1
        self.biases = np.random.rand(hidden_size)*2-1

        H = self.hidden_nodes(self.data)
        self.output_weights = np.dot(np.linalg.pinv(H), self.labels)            
  
    def hidden_nodes(self,X):
        G = np.dot(X, self.input_weights)
        G = G + self.biases
        H=self.activation(G)
        return H
  
    def predict(self,X):
        out = self.hidden_nodes(X)
        out = np.dot(out, self.output_weights)
        return out        
            
    def relu(self,x):
       return np.maximum(x, 0, x)

    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))
            
def normalize(A,option): #normalizing an array
    _,m=A.shape
    
    #lower and upper bounds joint angles
    lb_theta = np.array([0,-30,-30,-180,-90,-180])*np.pi/180;   
    ub_theta = np.array([90,60,30,180,90,180])*np.pi/180;     
    df_theta = ub_theta-lb_theta
    
    #lower and upper bounds force
    lb_force = np.array([-2000,-2000,-2000,0,0,0]); ub_force = np.array([2000,2000,2000,0,0,0]); df_force=ub_force-lb_force
    
    #lower and upper bounds of absolute pose xyz coordinates
    lb_pose = np.array([-1,-1,-1]); ub_pose = np.array([3,3,3]); df_pose=ub_pose-lb_pose          
    
    #lower and upper bounds of quaternions
    lb_quat=np.array([-1,-1,-1,-1]); ub_quat=np.array([1,1,1,1]); df_quat=ub_quat-lb_quat
    
    # relative pose and angular error shall not be normalized
    offset_z = np.array([0,0,0]); length_z=np.array([1,1,1])
    
    if (option=="minmax"):
        offset=[np.min(A[:,i:i+1]) for i in range(m)]
        length=[np.max(A[:,i:i+1])-np.min(A[:,i:i+1]) for i in range(m)]
    
    elif(option=="standard"):
        offset=[np.mean(A[:,i:i+1]) for i in range(m)]
        length=[np.std(A[:,i:i+1]) for i in range(m)]        
    
    elif(option=="dataset1"): #dataset 1 
        # input (6): six joint angles (rad); 
        # output (7): absolute pose xyz, i.e. position (m) and quaternions
        offset=np.concatenate((lb_theta, lb_pose,lb_quat))
        length=np.concatenate((df_theta, df_pose, df_quat))
    
    elif(option=="dataset2"): #dataset 2 
        # input (12): six joint angles (rad), wrench vector (force(N), torque(Nm)=zeros)
        # output (7): absolute pose, i.e. position (m) and quaternions
        offset = np.concatenate((lb_theta, lb_force, lb_pose, lb_quat))
        length = np.concatenate((df_theta, df_force, df_pose, df_quat))        

    elif(option=="dataset3"): #dataset 3 
        # input (12): six joint angles (rad), wrench vector (force(N), torque(Nm)=zeros)
        # output (7): relative pose, i.e. position error (m) and quaternion error
        offset = np.concatenate((lb_theta, lb_force, offset_z, lb_quat))
        length = np.concatenate((df_theta, df_force, length_z, df_quat))                
        
    elif(option=="dataset4"): #dataset 4 
        # input (6): six joint angles (rad)
        # intput (7): absolute pose, i.e. position (m) and quaternions
        offset = np.concatenate((lb_theta, lb_pose, lb_quat))
        length = np.concatenate((df_theta, df_pose, df_quat))            
        
    elif(option=="dataset5"): #dataset 5 
        # input (6): six joint angles (rad)
        # output (6): relative pose, i.e. position error (m) and angular error (rad)
        offset = np.concatenate((lb_theta, offset_z, offset_z))
        length = np.concatenate((df_theta, length_z, length_z))

    elif(option=="dataset6"): #dataset 6 
        # input (9): six joint angles (rad), force vector (N)
        # output (6): absolute pose, i.e. position (m) and angular error (rad)
        offset=np.concatenate((lb_theta, lb_force[:3],lb_pose,offset_z))
        length=np.concatenate((df_theta, df_force[:3],df_pose,length_z))        
    
    elif(option=="none"): #no normalization
        offset=np.zeros(m)
        length=np.zeros(m)
        
    else:
        print("Error: unknown option for normalization")                
        raise Exception("")
        
    for i in range(m):
        if(length[i]!=0):
            A[:,i:i+1] = (A[:,i:i+1]-offset[i])/length[i]        

    return(offset,length)

        
def denormalize(A,offset,length): #normalizing an array
    _,m=A.shape
    for i in range(m):
        if(length[i]!=0):
            A[:,i:i+1] = A[:,i:i+1]*length[i] + offset[i]
    
    
def error_prediction(filename, inputs=9, normalization="standard", neurons=100, activation="relu",xlim=10,n_bins=400):
    dataframe = np.loadtxt(filename,delimiter=",")
    m,N=dataframe.shape
    
    #normalization
    offset, length = normalize(dataframe,normalization)
    
    #data splitting
    data = dataframe[:,:inputs]
    labels= dataframe[:,inputs:]
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)

    #create model
    model=ELM_model(data_train,labels_train,neurons,activation)

    #predict
    prediction=model.predict(data_test)

    #denormalization    
    denormalize(prediction, offset, length)
    denormalize(labels_test, offset, length)

    return(prediction-labels_test)    

def print_prediction(filename, inputs=9, normalization="standard", neurons=100, activation="relu",xlim=10,n_bins=400):
    
    diff = error_prediction(filename, inputs, normalization, neurons, activation,xlim,n_bins)                                   
    
    # Convert from m in mm
    diff=diff*1000
    
    figure,axis=plt.subplots(1,3)
    figure.set_figwidth(50)
    figure.set_figheight(20)        
    maximum={}
    mean={}
    stderiv={}
    XLabels=["X error in mm", "Y error in mm", "Z error in mm"]    
    for i in range(3):        
        maximum[i]=np.max(abs(diff[:,i:i+1]))
        mean[i]=np.mean(diff[:,i:i+1])
        stderiv[i]=np.std(diff[:,i:i+1])
        axis[i].hist(abs(diff[:,i:i+1]),n_bins)
        axis[i].set_xlim(0,xlim)        
        axis[i].set_xlabel(XLabels[i])        
    plt.show()     
    label=["X","Y","Z"]
    print('{:<25}{:^25}{:>25}'.format("Maximum","Mean","StDerv"))
    print('{:<25}{:^25}{:>25}'.format("-----","-----","-----"))    
    for i in range(3):
        print(label[i],'{:<25}{:^25}{:>25}'.format(str(maximum[i]),str(mean[i]),str(stderiv[i])))
        
    absolute=[np.sqrt(diff[i,0]**2 + diff[i,1]**2 + diff[i,2]**2) for i in range(len(diff))]
    print("Average absolute positional error in mm: ", np.average(absolute))

def loss(filename, inputs=9, normalization="standard", neurons=100, activation="relu",xlim=10,n_bins=400):
    diff = error_prediction(filename, inputs, normalization, neurons, activation,xlim,n_bins)*1000
    absolute=[np.sqrt(diff[i,0]**2 + diff[i,1]**2 + diff[i,2]**2) for i in range(len(diff))]
    return(np.mean(absolute))


# In[49]:



print_prediction("datasets/dataset4.csv",inputs=6,normalization="minmax",neurons=234,activation="sigmoid",xlim=0.5)  



a=30
L4=[loss("datasets/dataset4.csv",inputs=6,normalization="dataset4",neurons=x,activation="sigmoid") for x in range(a,300)]
minloss=min(L4)
neurons=L4.index(minloss)+a
print(neurons,minloss)
print_prediction("datasets/dataset4.csv", inputs=6,normalization="dataset4",neurons=neurons,activation="sigmoid",xlim=0.3) 
 


# In[61]:


def searchTheBest(i,a,b=300,size=1):
    Datasets_input=[0,6,12,12,6,6,9]
    inputs=Datasets_input[i]
    print("Looking at Dataset",i," with ",inputs, " input neurons.")
    dataset="dataset"+str(i)
    filename="datasets/"+dataset+".csv"
    L=[loss(filename,inputs=inputs,normalization=dataset,neurons=x,activation="sigmoid") for x in range(a,b)]
    minloss=min(L)
    neurons=L.index(minloss)+a
    print("Neurons:",neurons," Loss: ", minloss)
    print_prediction(filename,inputs=inputs,normalization=dataset,neurons=neurons,activation="sigmoid",xlim=size)        


# In[64]:


searchTheBest(4,70,size=0.1)


# In[87]:


# In[66]:


searchTheBest(5,30,size=0.5)


# In[72]:


searchTheBest(6,20,b=400,size=0.2)


# In[73]:


searchTheBest(3,10,size=1)


# In[ ]:




