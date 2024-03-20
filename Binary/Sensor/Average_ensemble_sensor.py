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


# In[7]:


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
print('DT')

clf_2 = RandomForestClassifier(criterion="gini",max_depth=50,n_estimators=100)
clf_2.fit(X_train_scaled,y_train)
predictions_2=clf_2.predict(X_test_scaled)

print('RF')
# In[5]:


clf_3 = KNeighborsClassifier()
clf_3.fit(X_train_scaled,y_train)
predictions_3=clf_3.predict(X_test_scaled)

print('KNN')
# In[ ]:


from sklearn.svm import SVC
clf_4 = SVC(kernel='rbf', C=1, gamma='auto', probability=True)
clf_4.fit(X_train_scaled, y_train)
predictions_4=clf_4.predict(X_test_scaled)

print('SVM')
# In[6]:



from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


clf_5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=50),n_estimators=200)
clf_5.fit(X_train_scaled, y_train)
predictions_5=clf_5.predict(X_test_scaled)

print('AdaBoost')

# In[7]:


clf_6 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
clf_6.fit(X_train_scaled, y_train)
predictions_6=clf_6.predict(X_test_scaled)

print('NN')
# In[9]:


import pandas as pd

# Create a DataFrame to store predictions
predictions_df = pd.DataFrame({
    'DecisionTree': predictions_1,
    'RandomForest': predictions_2,
    'KNeighbors': predictions_3,
    'AdaBoost': predictions_5,
    'MLP': predictions_6
})


# In[12]:


y_test


# In[10]:


predictions_df


# In[8]:


# Sum the values along each row
row_sums = predictions_df.sum(axis=1)

# Apply threshold to convert sums into binary values
ensemble_predictions = (row_sums > 3).astype(int)

# Now, ensemble_predictions contains the final ensemble prediction
# You can use it for evaluation or any further processing


# In[14]:


ensemble_predictions


# In[9]:





# In[15]:


from sklearn.metrics import classification_report
print(classification_report(y_test,ensemble_predictions))


# In[16]:


from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, ensemble_predictions)

# Print accuracy
print("Accuracy:", accuracy)

