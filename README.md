# Neural-Network-Model-Building-DeepLearning
## AIM
### To develop a neural network classification model for the given dataset.

## Problem Statement

Write a python code to convert the categorical input to numeric values and  convert the categorical output to numeric values.  Build a TensorFlow model with an appropriate activation function and the number of neurons in the output layer. Draw the neural network architecture for your model using the following website
http://alexlenail.me/NN-SVG/index.html
## Neural Network Model
![image](https://github.com/gpavithra673/Neural-Network-Model-Building-DeepLearning/assets/93427264/c8a1fa24-80ff-4e6d-aa06-c688c311f376)

## DESIGN STEPS

### STEP 1:Import the necessary packages & modules

### STEP 2:Load and read the dataset

### STEP 3:Perform pre processing and clean the dataset

### STEP 4:Encode categorical value into numerical values using ordinal/label/one hot encoding

### STEP 5:Visualize the data using different plots in seaborn

### STEP 6:Normalize the values and split the values for x and y

### STEP 7:Build the deep learning model with appropriate layers and depth

### STEP 8:Analyze the model using different metrics

### STEP 9:Plot a graph for Training Loss, Validation Loss Vs Iteration & for Accuracy, Validation Accuracy vs Iteration

### STEP 10:Display the graph.
## PROGRAM

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import classification_report as report
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix as conf

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Read the data

df = pd.read_csv("mushrooms.csv")
df.columns
df.dtypes
df.shape
df.isnull().sum()
df_cleaned = df.dropna(axis=0)
df_cleaned.isnull().sum()
df_cleaned.shape
df_cleaned.dtypes

df_cleaned['class'].unique()
df_cleaned['cap-shape'].unique()
df_cleaned['cap-surface'].unique()
df_cleaned['cap-color'].unique()
df_cleaned['bruises'].unique()
df_cleaned['odor'].unique()
df_cleaned['gill-attachment'].unique()
df_cleaned['gill-spacing'].unique()
df_cleaned['gill-size'].unique()
df_cleaned['gill-color'].unique()
df_cleaned['stalk-shape'].unique()
 df_cleaned['stalk-root'].unique() 
df_cleaned['stalk-surface-above-ring'].unique()
df_cleaned['stalk-surface-below-ring'].unique()
df_cleaned['stalk-color-above-ring'].unique()
df_cleaned['stalk-color-below-ring'].unique()
df_cleaned['veil-type'].unique()
df_cleaned['veil-color'].unique()
df_cleaned['ring-number'].unique()
df_cleaned['ring-type'].unique()
df_cleaned['spore-print-color'].unique()
df_cleaned['population'].unique()
df_cleaned['habitat'].unique()

categories_list=[['p', 'e'],['x', 'b', 's', 'f', 'k', 'c'],['s', 'y', 'f', 'g'],
                 ['n', 'y', 'w', 'g', 'e', 'p', 'b', 'u', 'c', 'r'],['t', 'f'],['p', 'a', 'l', 'n', 'f', 'c', 'y', 's', 'm'],['f', 'a'],['c', 'w'],['n', 'b'],
                 ['k', 'n', 'g', 'p', 'w', 'h', 'u', 'e', 'b', 'r', 'y', 'o'],['e', 't'],['e', 'c', 'b', 'r', '?'],['s', 'f', 'k', 'y'],['s', 'f', 'y', 'k'],
                 ['w', 'g', 'p', 'n', 'b', 'e', 'o', 'c', 'y'],['w', 'p', 'g', 'b', 'n', 'e', 'y', 'o', 'c'],['p'],['w', 'n', 'o', 'y'],['o', 't', 'n'],
                 ['p', 'e', 'l', 'f', 'n'],['k', 'n', 'u', 'h', 'w', 'r', 'o', 'y', 'b'],['s', 'n', 'a', 'v', 'y', 'c']]
enc = OrdinalEncoder(categories=categories_list)

df1 = df_cleaned.copy()

df1[['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
     'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
     'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population']] = enc.fit_transform(df1[['class','cap-shape'
     ,'cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
     'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
     'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population']])
df1
df1.dtypes

le = LabelEncoder()
df1['habitat'] = le.fit_transform(df1['habitat'])
df1.dtypes

x = df1[['class','cap-shape'
     ,'cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
     'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
     'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population']].values
         
y1 = df1[['habitat']].values

ohe = OneHotEncoder()
ohe.fit(y1)

y = ohe.transform(y1).toarray()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=50)
# Build the Model
# Build the Model

ai = Sequential([Dense(77,input_shape = [22]),
                 Dense(77,activation="relu"),
                 Dense(65,activation="relu"),
                 Dense(45,activation="relu"),
                 Dense(7,activation="softmax")])

ai.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='max', 
    verbose=1, 
    patience=20)
    
ai.fit( x = x_train, y = y_train,
        epochs=500, batch_size=256,
        validation_data=(x_test,y_test),
        callbacks = [early_stop]
        ) 
# Analyze the model

metrics = pd.DataFrame(ai.history.history)
metrics.head()

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

x_pred = np.argmax(ai.predict(x_test), axis=1)
x_pred.shape

y_truevalue = np.argmax(y_test,axis=1)
y_truevalue.shape

conf(y_truevalue,x_pred)

print(report(y_truevalue,x_pred))


```
## OUTPUT
### DATA information:
![image](https://github.com/gpavithra673/Neural-Network-Model-Building-DeepLearning/assets/93427264/3057757f-37d2-4feb-8be7-2b267fc9f239)


### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/gpavithra673/Neural-Network-Model-Building-DeepLearning/assets/93427264/56e4c0ae-de86-4207-ad55-3066f0f026cf)

### Classification Report

![image](https://github.com/gpavithra673/Neural-Network-Model-Building-DeepLearning/assets/93427264/816a4ec1-5c4e-4ddd-bc6b-c4b47b398b42)

## RESULT:
### A neural network classification model is developed for the given dataset.
