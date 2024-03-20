#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

#!/usr/bin/env python
# coding: utf-8

# In[6]:




df=pd.read_csv('labeled_dataset_CPSS.csv')


# In[12]:


df.head()


# In[13]:


df.shape


# In[15]:


# In[2]:

df.head()
data_df=df
data_df.head()





from sklearn.preprocessing import StandardScaler

# In[20]:


from sklearn.model_selection import train_test_split

# define the feature columns and target variable
feature_cols = ['Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time']
X = df[feature_cols] # Features
y = df.Label # Target variable

from sklearn.model_selection import train_test_split


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[2]:


from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier


# In[4]:


estimators = []

# AdaBoostClassifier with DecisionTreeClassifier as base estimator
estimators.append(('AdaBoostClassifier', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4), n_estimators=200)))

# KNeighborsClassifier
estimators.append(('KNN', KNeighborsClassifier()))

# MLPClassifier
estimators.append((('MLPClassifier'(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)))

# Decision Tree Classifier
estimators.append((('Decision Tree Classifier', DecisionTreeClassifier(random_state=13)))

# RandomForestClassifier
estimators.append(('RandomForest', RandomForestClassifier(criterion="gini", max_depth=4, n_estimators=100)))

# SVC
estimators.append(('SVC', SVC(random_state=13)))


# In[5]:


from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

estimators = []

# AdaBoostClassifier with DecisionTreeClassifier as base estimator
estimators.append(('AdaBoostClassifier', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4), n_estimators=200)))

# KNeighborsClassifier
estimators.append(('KNN', KNeighborsClassifier()))

# MLPClassifier
estimators.append(('MLPClassifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)))

# Decision Tree Classifier
estimators.append(('Decision Tree Classifier', DecisionTreeClassifier(random_state=13)))

# RandomForestClassifier
estimators.append(('RandomForest', RandomForestClassifier(criterion="gini", max_depth=4, n_estimators=100)))

# SVC
estimators.append(('SVC', SVC(random_state=13)))


# In[6]:


from xgboost import XGBClassifier

# Create XGBClassifier
XGB = XGBClassifier(random_state=13)


# In[7]:


from sklearn.ensemble import StackingClassifier
SC = StackingClassifier(estimators=estimators,final_estimator=XGB,cv=6)
SC.fit(X_train_scaled, y_train)
predictions = SC.predict(X_test_scaled)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

