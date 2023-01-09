#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


import numpy as np
dataframe = np.loadtxt("modeldata.csv",delimiter=",")


# In[3]:


from sklearn.model_selection import train_test_split

data = dataframe[:,:9]
labels= dataframe[ :,9:]

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)

data_train.shape, labels_train.shape


# In[4]:


model = Sequential()
model.add(Dense(100, input_dim=9, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(7, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(data_train,labels_train, epochs=100, batch_size=20, validation_data=(data_test, labels_test))


# In[ ]:




