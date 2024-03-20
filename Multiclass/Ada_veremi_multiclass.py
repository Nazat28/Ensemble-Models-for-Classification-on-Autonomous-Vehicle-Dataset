#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt


# In[4]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df=pd.read_csv('merged_modified.csv')


# In[2]:


# drop the columns 'pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z'
# drop the columns 'pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z'
# df = df.drop([ 'pos_y' , 'pos_x','spd_x','pos_z','spd_z'], axis=1)
data_df=df

df.head()


# In[3]:


from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = [ 'pos_y' , 'pos_x','spd_x','pos_z','spd_z','spd_y']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable

from sklearn.model_selection import train_test_split


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[6]:


base_estimator = DecisionTreeClassifier(max_depth=10)
classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50)


classifier.fit(X_train_scaled, y_train)


predictions = classifier.predict(X_test_scaled)


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

