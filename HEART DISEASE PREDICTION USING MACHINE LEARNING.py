#!/usr/bin/env python
# coding: utf-8

# IMPORTING THE DEPENDENCIES

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# DATA COLLECTION AND PROCESSING

# In[2]:


#loading the csv data to a pandas dataframe
heart_data = pd.read_csv(r"C:\Users\ISHABIYI SIMON\Downloads\heart_disease_data.csv")


# In[3]:


# print the first 5 rows of the dataset
heart_data.head()


# In[4]:


# print last 5 rows of the data
heart_data.tail()


# In[5]:


#number of rows  and columns in the dataset
heart_data.shape


# In[6]:


#getting some information aboy the dataset
heart_data.info()


# In[7]:


#checking for missing values
heart_data.isnull().sum()


# In[8]:


#statistical measures about the data
heart_data.describe()


# In[9]:


# checking the distribution of the target variable
heart_data['target'].value_counts()


#  1= defective heart
#  0= healty heart

# SPLITTING THE FEATURES AND TARGET

# In[10]:


X = heart_data.drop(columns= 'target', axis = 1)
Y = heart_data['target']


# In[11]:


print(X)


# In[12]:


print(Y)


# SPLITTING INTO TRAINING DATA AND TEST DATA

# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2 ,stratify = Y,random_state = 2)


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[17]:


print(X.shape, X_train.shape, X_test.shape)


# MODEL TRAINING DATA

# Logistic Regression

# In[19]:


model = LogisticRegression()


# In[20]:


# training the logistic Regression model
model.fit(X_train, Y_train)


# MODEL EVALUATION

# ACCURACY SCORE

# In[23]:


#Accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[24]:


print('Accuracy on training data:', training_data_accuracy)


# In[26]:


#Accuracy on the training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[27]:


print('Accuracy on test data:', test_data_accuracy)


# BUILDING A PREDICT SYSTEM

# In[30]:


input_data = (58,0,0,170,225,1,0,146,1,2.8,1,2,1)

#changing the input_data to an numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

print(prediction)
if (prediction[0]==0):
  print('The Person does not have Heart Disease')
else:
  print('The Person has Heart Disease')


# In[ ]:




