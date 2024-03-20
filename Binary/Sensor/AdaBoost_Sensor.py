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

df=pd.read_csv('labeled_dataset_CPSS.csv')


# In[2]:


# drop the columns 'Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time'
data_df=df
# df = df.drop(['Consistency', 'Location', 'Lane Alignment', 'Correlation', 'Speed','Headway Time', 'Plausibility','Formality'], axis=1)
df.head()


# In[3]:


from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time']
X = data_df[feature_cols] # Features
y = data_df.Label # Target variable

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


base_estimator = DecisionTreeClassifier(max_depth=4)
clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50)


clf.fit(X_train_scaled, y_train)


predictions = clf.predict(X_test_scaled)


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

