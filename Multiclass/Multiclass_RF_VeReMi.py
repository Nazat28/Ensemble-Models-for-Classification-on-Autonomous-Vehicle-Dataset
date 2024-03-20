#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
df=pd.read_csv('merged_modified.csv')


# In[2]:


feature_cols = ['pos_x','pos_y','pos_z','spd_x','spd_y','spd_z']

X = df[feature_cols] # Features
y = df.attackerType # Target variable


# In[3]:


df.tail()


# In[7]:


from sklearn.model_selection import train_test_split

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import StandardScaler

# create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[10]:


from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(criterion="gini",max_depth=50,n_estimators=200)
clf.fit(X_train_scaled,y_train)


# In[6]:


predictions= clf.predict(X_test_scaled)


# In[8]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[6]:


import shap

# Fits the explainer
explainer = shap.Explainer(clf.predict, X_test_scaled)
start_index = 0
end_index = 1000
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test_scaled[start_index:end_index])


# In[7]:


# Define a list of feature names
features = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
from scipy.special import softmax

def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")


# In[8]:


# Print the feature importances based on the SHAP values
print_feature_importances_shap_values(shap_values, features)


# In[ ]:


import time
import shap
import matplotlib.pyplot as plt

start = time.time()

start_index = 0
end_index = 50000

# Your existing code for creating the explainer and computing SHAP values
explainer = shap.KernelExplainer(clf.predict_proba, X_test_scaled[start_index:end_index])
shap_values = explainer.shap_values(X_test_scaled[start_index:end_index])

# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']

# Generate the summary plot using matplotlib
shap.summary_plot(shap_values=shap_values,
                  features=X_test_scaled[start_index:end_index],
                  feature_names=feature_cols,
                  class_names=['Normal', 'Constant', 'Constant Offset',
                               'Random', 'Random Offset', 'Eventual Stop'],
                  show=False)

# Get the current axis (assuming it's the only axis in the plot)
ax = plt.gca()

# Move the legend to the desired location (e.g., lower-right)
ax.legend(loc='lower right')

# Show the plot
plt.show()

# END timer
end = time.time()
print('SHAP time for RF: ',(end - start), 'sec')


# In[23]:


shap.summary_plot(shap_values = shap_values[5],features = X_test_scaled[start_index:end_index], feature_names= feature_cols,show=False)


# In[6]:


import lime
import lime.lime_tabular

# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
# Get the class names
class_names=['Normal (0)', 'Constant (1)', 'Constant Offset (2)',
                               'Random (4)', 'Random Offset(8)', 'Eventual Stop (16)']

# Get the feature names
feature_names = list(feature_cols)

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train_scaled),
                    feature_names=feature_names, 
                    class_names=class_names,                          
                    verbose=True, mode='classification')


# In[9]:


import time

start = time.time()

for i in range(50000):
  explanation = explainer.explain_instance(X_test_scaled[i], clf.predict_proba)

end = time.time()

print('Total LIME time for 1000 samples:', (end - start), 'sec')


# In[84]:


import numpy as np

# Indices where predictions don't match with y_test
mismatch_indices = np.where(predictions != y_test)[0]

print(mismatch_indices)


# In[89]:


instance_idx=2
print(y_test.iloc[instance_idx])
print(predictions[instance_idx])


# In[90]:


import numpy as np
from lime import lime_tabular

start = time.time()

# Show the result of the model's explanation
explanation = explainer.explain_instance(X_test_scaled[instance_idx], clf.predict_proba)
explanation.show_in_notebook(show_table=True, show_all=False)

# END timer
end = time.time()
print('SHAP time for RF: ',(end - start), 'sec')


# In[43]:


import pandas as pd

# Assuming X_test_scaled_array is the 1D array containing the scaled values
X_test_scaled_array = X_test_scaled[instance_idx]

# Assuming feature_cols contains the names of the feature columns
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']

# Create a pandas DataFrame with a single row
scaled_data_df = pd.DataFrame([X_test_scaled_array], columns=feature_cols)

# Print the DataFrame
print(scaled_data_df)


# In[17]:


import numpy as np
import matplotlib.pyplot as plt

# Dictionary to store sparsity values
sparsity_dict = {}

# Features
features = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']

# Loop through SHAP values for each class
for class_shap_values in shap_values:

  # Convert to numpy array
  class_array = np.array(class_shap_values)

  # Loop through features
  for i in range(class_array.shape[1]): 

    # Get SHAP values for feature
    feature_shap = class_array[:, i]
    
    # Get feature name
    feat_name = features[i]

    # Create histogram
    histo = plt.hist(feature_shap, bins=20)

    # Get bin edges and convert to array
    bin_edges = np.array(plt.gca().get_xbound())

    # Index of bin near zero
    zero_bin_idx = np.argmin(np.abs(bin_edges - 0))

    # Get bin heights
    zero_bin_counts = histo[0][zero_bin_idx]

    # Total number of values
    total_count = len(feature_shap)

    # Calculate sparsity
    sparsity = zero_bin_counts / total_count

    # Store sparsity for feature in dictionary
    sparsity_dict[feat_name] = sparsity
    
    # Clear plot
    plt.clf()

# Print sparsity for each feature  
for feat in features:
  print(f"{feat} -> {sparsity_dict[feat]:.3f}")


# In[18]:


# Histogram of SHAP values
counts, bins = np.histogram(shap_values, bins=20)

# Find index of bin closest to zero  
zero_bin_idx = np.argmin(np.abs(bins))  

# Get counts in zero bin
zero_bin_counts = counts[zero_bin_idx]

# Calculate sparsity
sparsity = zero_bin_counts / sum(counts)
print(sparsity)


# In[19]:


from lime import lime_tabular
import numpy as np

start_index = 0
end_index = 1000


# Initialize LimeTabularExplainer outside the loop
explainer = lime_tabular.LimeTabularExplainer(X_test_scaled, mode='classification', feature_names=feature_names)

# Calculate LIME explanations for the instances of interest
lime_explanations = []
for instance in X_test_scaled[start_index:end_index]:
    explanation = explainer.explain_instance(instance, clf.predict_proba, num_features=len(X_test_scaled[0]))
    lime_explanations.append(explanation)

# Extract the local interpretable models and feature importances
local_models = [exp.local_exp[1] for exp in lime_explanations]

# Calculate sparsity
sparsity_values = []
for local_model in local_models:
    feature_importances = np.array([val for _, val in local_model])
    num_zero_coefficients = np.sum(np.abs(feature_importances) < 1e-10)
    sparsity = num_zero_coefficients / len(feature_importances)
    sparsity_values.append(sparsity)

# Average sparsity over all instances
average_sparsity = np.mean(sparsity_values)
print(average_sparsity) 

