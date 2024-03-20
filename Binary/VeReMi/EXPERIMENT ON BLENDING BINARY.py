#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[11]:


df=pd.read_csv('modified_data.csv')

df.head()

df.shape


# In[2]:


# drop the columns 'pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z'

#df = df.drop([  'pos_y' ,'pos_x','spd_x','pos_z','spd_z'], axis=1)
data_df=df
df.head()


# In[3]:


from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = [  'pos_y' ,'pos_x','spd_x','pos_z','spd_z','spd_y']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable


# In[13]:


from sklearn.model_selection import train_test_split


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=100)

# create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[14]:


X_train_scaled


# In[15]:


from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report


# In[16]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier


# In[20]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

estimators = []

# AdaBoostClassifier with DecisionTreeClassifier as base estimator
estimators.append(('AdaBoostClassifier', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4), n_estimators=200)))

# KNeighborsClassifier
estimators.append(('KNN', KNeighborsClassifier()))

# MLPClassifier
estimators.append(('MLPClassifier', MLPClassifier(max_iter=2000, random_state=13)))

# Decision Tree Classifier
estimators.append(('Decision Tree Classifier', DecisionTreeClassifier(random_state=13)))

# RandomForestClassifier
estimators.append(('RandomForest', RandomForestClassifier(criterion="gini", max_depth=4, n_estimators=100)))

# SVC
# estimators.append(('SVC', SVC(random_state=13)))


# In[8]:


from xgboost import XGBClassifier

# Create XGBClassifier
XGB = XGBClassifier(random_state=13)


# In[5]:


# Experimenting if it works on 50K samples
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report

# Slice the first 50,000 data points for training and testing
X_train_subset = X_train_scaled[:50000]
y_train_subset = y_train[:50000]
X_test_subset = X_test_scaled[:50000]
y_test_subset = y_test[:50000]

# Create XGBClassifier
XGB = XGBClassifier(random_state=13)

# Create StackingClassifier
SC = StackingClassifier(estimators=estimators, final_estimator=XGB, cv=6)

# Fit the StackingClassifier on the subset of training data
SC.fit(X_train_subset, y_train_subset)

# Make predictions on the subset of testing data
predictions = SC.predict(X_test_subset)

# Print classification report
print(classification_report(y_test_subset, predictions))


# In[9]:


from xgboost import XGBClassifier

# Create XGBClassifier
XGB = XGBClassifier(random_state=13)


# In[ ]:


from sklearn.ensemble import StackingClassifier
SC = StackingClassifier(estimators=estimators,final_estimator=XGB)
SC.fit(X_train_scaled, y_train)
predictions = SC.predict(X_test_scaled)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

