#!/usr/bin/env python
# coding: utf-8

# # Iris Dataset task

# 
# - The task using the Iris data set, it would need to preprocess the data, develop and train a Deep Learning model, and evaluate the performance of the model. 
# 
# 
# - Developing and training the model could involve selecting an appropriate architecture and optimization algorithm, setting the learning rate, and choosing the number of epochs. 
# 
# - Evaluating the performance of the model could involve using metrics such as accuracy, precision, and recall to assess the model's ability to classify the iris plants correctly.

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[2]:


# Step1: Load the dataset

iris_data = pd.read_csv("IRIS.csv")


# In[3]:


# Step 2: define features and labels

X = iris_data.drop('species', axis=1).values
y = pd.Categorical(iris_data['species']).codes


# In[4]:


# Step 3: Shuffle the data

np.random.seed(1)

shuffled_indices = np.arange(len(X)) 
np.random.shuffle(shuffled_indices)

X = X[shuffled_indices]
y = y[shuffled_indices]


# In[5]:


# Step 4: Splitting the dataset into training and testing sets 

split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# In[6]:


# Step 5: Building the model 

model = Sequential([
    Dense(8, input_dim=4, activation='relu'),  
    Dense(3, activation='relu'), 
    Dense(3, activation='softmax')  # the output layer has 3 neuronsf or three classes
])


# In[7]:


# Step 6: Compile the model 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[8]:


# Step 7: Test the model with a small sample of the data

sample_size = 10
X_sample, y_sample = X_train[:sample_size], y_train[:sample_size]


# In[9]:


# Step 8: Ensure the model works before training

model_output = model.predict(X_sample)
print("Model output, before training:")
print(model_output)


# In[10]:


# Step 9: Train the model

model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)


# In[11]:


# Step 10: Evaluate the model on the test set

accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"\nThe accuracy on the test set: {accuracy}")

