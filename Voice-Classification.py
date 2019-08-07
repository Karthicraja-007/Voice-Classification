#!/usr/bin/env python
# coding: utf-8

# ## Gender Recognition by Voice and Speech Analysis

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[5]:


data = pd.read_csv(r'C:\Users\Ramkumar\Downloads\Raja\Projects\voice.csv')


# In[6]:


data.head()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.isnull().sum()


# In[14]:


print(" Shape of Data : " ,data.shape)
print(" Total no of labels : {}".format(data.shape[0]))
print(" No of male : {}".format(data[data.label == 'male'].shape[0]))
print(" No of Female : {}".format(data[data.label == 'female'].shape[0]))


# In[15]:


X = data.iloc[:, :-1]
y = data.iloc[:, -1:]


# In[16]:


print(X.shape)
print(y.shape)


# In[18]:


from sklearn.preprocessing import LabelEncoder
lab_en = LabelEncoder()

y = lab_en.fit_transform(y)


# In[19]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[34]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)


# # Logistic Regression

# In[35]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


# In[36]:


from sklearn import metrics
print("Logistic Regression Test Accuracy = {}".format(lr.score(X_test, y_test)))


# # Support Vector Model

# In[37]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)


# In[38]:


print("Accuracy score : {}".format(metrics.accuracy_score(y_test, y_pred_svc)))


# In[39]:


print(confusion_matrix(y_test, y_pred_svc))


# # Naive Bayes

# In[40]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

print("Accuracy score of Naive Bayes : {}".format(nb.score(X_test, y_test)))


# # KNN

# In[41]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train)

print ("Accuracy score of KNN : {}".format(knn.score(X_test, y_test)))


# # Decision Tree model

# In[42]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=10, min_samples_leaf = 15)
dtree.fit(X_train, y_train)

print("Accuracy Score of Decision Tree : {}".format(dtree.score(X_test, y_test)))


# # Random Forest Model

# In[44]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 70)
rf.fit(X_train, y_train)

print("Accuracy score of Random forest : {}".format(rf.score(X_test, y_test)))


# # SGD Model

# In[45]:


from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss = 'modified_huber', shuffle = True)
sgd.fit(X_train, y_train)

print("Accuracy score of SGD : {}".format(sgd.score(X_test, y_test)))


# In[46]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

print("Shape of Scaled Train set : ", scaled_X_train.shape)


# # Tensorflow and Keras

# In[51]:


import tensorflow as tf

from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import losses, metrics, optimizers


# In[53]:


dnn_keras_model = models.Sequential()


# In[54]:


dnn_keras_model.add(layers.Dense(units=30, input_dim=20, activation = 'relu'))
dnn_keras_model.add(layers.Dense(units=20, activation = 'relu'))
dnn_keras_model.add(layers.Dense(units=10, activation = 'relu'))
dnn_keras_model.add(layers.Dense(units=2, activation = 'softmax'))


# In[56]:


#Compile model by selecting optimizer and loss function
dnn_keras_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[57]:


dnn_keras_model.fit(scaled_X_train, y_train, epochs = 50)


# In[59]:


keras_pred = dnn_keras_model.predict_classes(scaled_X_test)


# In[64]:


from sklearn.metrics import accuracy_score
print ("Accuracy score of Keras model : {}".format(accuracy_score(keras_pred, y_test)))


# In[ ]:




