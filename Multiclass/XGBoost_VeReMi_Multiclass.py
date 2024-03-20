#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:


# In[1]:


import pandas as pd
import numpy as np


# In[11]:

df=pd.read_csv('merged_modified.csv')


# In[12]:


df.head()


# In[13]:


df.shape


# In[15]:


# In[9]:


import pandas as pd


# Create a copy of the DataFrame
df_modified = df.copy()

 # Replace values in the first column where the value is 2 with 1 in the modified DataFrame
df_modified.loc[df_modified['attackerType'] == 2, 'attackerType'] = 2

# Replace values in the first column where the value is 8 with 4 in the modified DataFrame
df_modified.loc[df_modified['attackerType'] == 4, 'attackerType'] = 3

# Replace values in the first column where the value is 8 with 4 in the modified DataFrame
df_modified.loc[df_modified['attackerType'] == 8, 'attackerType'] = 4

# Replace values in the first column where the value is 16 with 5 in the modified DataFrame
df_modified.loc[df_modified['attackerType'] == 16, 'attackerType'] = 5

# Save the modified DataFrame to a new file
df_modified.to_csv("modified_dataset1.csv", index=False)


# In[10]:


# drop the columns , 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z'
df=pd.read_csv('modified_dataset1.csv')
# df = df.drop([ 'pos_z', 'spd_z', 'spd_x'], axis=1)
df.head()
data_df=df
data_df.head()


# In[11]:


from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['pos_x','spd_y','pos_y', 'pos_z', 'spd_z', 'spd_x' ]
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


# In[12]:


num_classes = len(set(y_train))


# In[13]:


num_classes


# In[14]:


import xgboost as xgb

# Create an XGBoost classifier
clf = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes, eval_metric='mlogloss')

# Train the classifier
clf.fit(X_train_scaled, y_train)

# Predict on the test set
predictions = clf.predict(X_test_scaled)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[4]:


import xgboost as xgb
# Create an XGBoost classifier
clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# Train the classifier
clf.fit(X_train_scaled, y_train)

# Predict on the test set
predictions = clf.predict(X_test_scaled)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

