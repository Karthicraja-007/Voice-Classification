#!/usr/bin/env python
# coding: utf-8

# ## Gender Recognition by Voice and Speech Analysis
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r'Input\voice.csv')

data.head()
data.info()
data.describe()
data.isnull().sum()

print(" Shape of Data : " ,data.shape)
print(" Total no of labels : {}".format(data.shape[0]))
print(" No of male : {}".format(data[data.label == 'male'].shape[0]))
print(" No of Female : {}".format(data[data.label == 'female'].shape[0]))

X = data.iloc[:, :-1]
y = data.iloc[:, -1:]

print(X.shape)
print(y.shape)

from sklearn.preprocessing import LabelEncoder
lab_en = LabelEncoder()
y = lab_en.fit_transform(y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

from sklearn import metrics
print("Logistic Regression Test Accuracy = {}".format(lr.score(X_test, y_test)))


# # Support Vector Model

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)

print("Accuracy score : {}".format(metrics.accuracy_score(y_test, y_pred_svc)))
print(confusion_matrix(y_test, y_pred_svc))


# # Naive Bayes

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

print("Accuracy score of Naive Bayes : {}".format(nb.score(X_test, y_test)))


# # KNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train)

print ("Accuracy score of KNN : {}".format(knn.score(X_test, y_test)))


# # Decision Tree model

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=10, min_samples_leaf = 15)
dtree.fit(X_train, y_train)

print("Accuracy Score of Decision Tree : {}".format(dtree.score(X_test, y_test)))


# # Random Forest Model

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 70)
rf.fit(X_train, y_train)

print("Accuracy score of Random forest : {}".format(rf.score(X_test, y_test)))


# # SGD Model

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss = 'modified_huber', shuffle = True)
sgd.fit(X_train, y_train)

print("Accuracy score of SGD : {}".format(sgd.score(X_test, y_test)))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

print("Shape of Scaled Train set : ", scaled_X_train.shape)


# # Tensorflow and Keras

import tensorflow as tf

from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import losses, metrics, optimizers

dnn_keras_model = models.Sequential()

dnn_keras_model.add(layers.Dense(units=30, input_dim=20, activation = 'relu'))
dnn_keras_model.add(layers.Dense(units=20, activation = 'relu'))
dnn_keras_model.add(layers.Dense(units=10, activation = 'relu'))
dnn_keras_model.add(layers.Dense(units=2, activation = 'softmax'))

#Compile model by selecting optimizer and loss function
dnn_keras_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

dnn_keras_model.fit(scaled_X_train, y_train, epochs = 50)

keras_pred = dnn_keras_model.predict_classes(scaled_X_test)

from sklearn.metrics import accuracy_score
print ("Accuracy score of Keras model : {}".format(accuracy_score(keras_pred, y_test)))

