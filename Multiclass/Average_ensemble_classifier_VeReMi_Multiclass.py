#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd

# load the dataset into a pandas DataFrame
df=pd.read_csv('merged_modified.csv')


# In[91]:


df.head()


# In[13]:


df.shape


# In[93]:


from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
X = df[feature_cols] # Features
y = df.attackerType # Target variable

from sklearn.model_selection import train_test_split


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[2]:


import pandas as pd
import numpy as np
#from statistics import *
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# In[3]:


from sklearn.tree import DecisionTreeClassifier

clf_1 = DecisionTreeClassifier(criterion="gini", max_depth=50)
clf_1.fit(X_train_scaled, y_train)
predictions_1 = clf_1.predict(X_test_scaled)


# In[4]:


clf_2 = RandomForestClassifier(criterion="gini",max_depth=50,n_estimators=100)
clf_2.fit(X_train_scaled,y_train)
predictions_2=clf_2.predict(X_test_scaled)


# In[5]:


clf_3 = KNeighborsClassifier()
clf_3.fit(X_train_scaled,y_train)
predictions_3=clf_3.predict(X_test_scaled)


# In[ ]:


from sklearn.svm import SVC
clf_4 = SVC(kernel='rbf', C=1, gamma='auto', probability=True)
clf_4.fit(X_train_scaled[:10000], y_train)
predictions_4=clf_4.predict(X_test_scaled)


# In[6]:


clf_5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=50),n_estimators=200)
clf_5.fit(X_train_scaled, y_train)
predictions_5=clf_5.predict(X_test_scaled)


# In[7]:


clf_6 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
clf_6.fit(X_train_scaled, y_train)
predictions_6=clf_6.predict(X_test_scaled)


# In[18]:


import pandas as pd

# Create a DataFrame to store predictions
predictions_df = pd.DataFrame({
    'DecisionTree': 0.2*predictions_1,
    'RandomForest': 0.3*predictions_2,
    'KNeighbors': 0.2*predictions_3,
#     'SVM' : predictions_4,
    'AdaBoost': 0.15*predictions_5,
    'MLP': 0.15*predictions_6
})


# In[19]:


y_test


# In[20]:


predictions_df


# In[21]:


# Sum the values along each row
row_sums = predictions_df.sum(axis=1)

# Apply threshold to convert sums into binary values
ensemble_predictions = (row_sums).astype(int)

# Now, ensemble_predictions contains the final ensemble prediction
# You can use it for evaluation or any further processing


# In[22]:


# Define your labels
labels = [0, 1, 2, 4, 8, 16]

# Map average values to the nearest label
ensemble_predictions_mapped = [min(labels, key=lambda x: abs(x - value)) for value in ensemble_predictions]

# Now ensemble_predictions_mapped contains the final ensemble prediction mapped to the nearest label
# You can use it for evaluation or any further processing


# In[23]:


ensemble_predictions_mapped


# In[24]:


from sklearn.metrics import classification_report
print(classification_report(y_test,ensemble_predictions_mapped))
from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, ensemble_predictions_mapped)

# Print accuracy
print("Accuracy:", accuracy)

