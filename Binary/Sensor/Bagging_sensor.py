#!/usr/bin/env python
# coding: utf-8

# In[4]:


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




from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

base_estimator = DecisionTreeClassifier(criterion="gini", max_depth=50, min_samples_leaf=4, random_state=1)
model = BaggingClassifier(base_estimator, n_estimators=200, random_state=1)

# Fit the model
model.fit(X_train_scaled, y_train)



# In[8]:


predictions=model.predict(X_test_scaled)


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

