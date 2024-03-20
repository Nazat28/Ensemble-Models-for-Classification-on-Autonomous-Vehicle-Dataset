#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:


# In[1]:


import pandas as pd
import numpy as np


# In[11]:


df=pd.read_csv('modified_data.csv')


# In[12]:


df.head()


# In[13]:


df.shape


# In[15]:


# In[2]:


# drop the columns , 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z'
# df = df.drop([ 'pos_z', 'spd_z'], axis=1)
df.head()
data_df=df
data_df.head()


# In[3]:


from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['pos_x','spd_y','pos_y', 'spd_x','pos_z', 'spd_z']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable

from sklearn.model_selection import train_test_split


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[13]:


import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

#converting the dataset into proper LGB format 
d_train=lgb.Dataset(X_train_scaled, label=y_train)#Specifying the parameter
params={}
params['learning_rate']=0.80
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='binary' #Binary target feature
params['metric']='binary_logloss' #metric for binary classification
params['max_depth']=10#train the model 
# params['num_leaves']=10
clf=lgb.train(params,d_train,100) #train the model on 100 epocs#prediction on the test set
predictions=clf.predict(X_test_scaled)

# Convert probabilities to binary predictions
binary_predictions = (predictions >= 0.5).astype(int)

# Now you can print the classification report
print(classification_report(y_test, binary_predictions))

