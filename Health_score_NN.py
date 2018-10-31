# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:42:56 2018

@author: tatsa02
"""

import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import random
from sklearn.model_selection import train_test_split
#import cross_validation from sklearn
#%matplotlib inline
#print (os.getcwd())
os.chdir('C:/Users/tatsa02/Documents/Workspace-Python/ipython-notebooks-master/ipython-notebooks-master/data')
print (os.getcwd())
path = os.getcwd() + '/HELATHSCORE_DATA.csv'
data = pd.read_csv(path,header='infer')
#data=random.shuffle(data)
#print(data)
DUMMYS=pd.get_dummies(data,
                      columns=['GEO','ACCOUNT_SEGMENT','INDUSTRY','GBU1','INSIDE_OUTSIDE','BUSINESS_TYPE','OPPORTUNITY_SOURCE'],prefix_sep='_')

#DUMMYS=np.asarray(DUMMYS)

#dataframe=pd.DataFrame(e, columns=['a']) 
DUMMYS=pd.DataFrame(DUMMYS) 

data =pd.concat([data,DUMMYS],axis=1)


# create training and testing vars
X_train, X_test = train_test_split(data, test_size=0.2,random_state=42)
print (X_train.shape)
print (X_test.shape)


X_train_1=X_train.iloc[:, 8:99]

#data1['TARGET']

data_predictors_train=X_train_1.drop(['TARGET','OPPORTUNITY_NUMBER'], axis=1)

data_target_train=X_train['TARGET']

data_target_train=data_target_train.iloc[:,1]

data_target_train=pd.DataFrame(data_target_train) 

#from sklearn.linear_model import LogisticRegression
#
#classifier = LogisticRegression(random_state=0)
#
#
#classifier.fit(data_predictors_train, data_target_train)

X_test_1=X_test.iloc[:, 8:99]

#data1['TARGET']

data_predictors_test=X_test_1.drop(['TARGET','OPPORTUNITY_NUMBER'], axis=1)

data_target_test=X_test_1['TARGET']

data_target_test=data_target_test.iloc[:,1]

data_target_test=pd.DataFrame(data_target_test)


#y_pred = classifier.predict(data_predictors_test)
#from sklearn.metrics import confusion_matrix
#confusion_matrix = confusion_matrix(data_target_test, y_pred)
#print(confusion_matrix)
#
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.
#      format(classifier.score(data_predictors_test, data_target_test)))
#
#from sklearn.metrics import classification_report
#print(classification_report(data_target_test, y_pred))

import numpy as np

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


#Variable initialization
epoch=5000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = data_predictors_train.shape[1] #number of features in data set
hiddenlayer_neurons = 10 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer


#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
bout=np.array(bout)

for i in range(epoch):

#Forward Propogation
 hidden_layer_input1=np.dot(data_predictors_train,wh)
 hidden_layer_input=hidden_layer_input1 + bh
 hiddenlayer_activations = sigmoid(hidden_layer_input)
 output_layer_input1=np.dot(hiddenlayer_activations,wout)
 output_layer_input= output_layer_input1+ bout
 output = sigmoid(output_layer_input)

#Backpropagation
 E = data_target_train-output
 slope_output_layer = derivatives_sigmoid(output)
 slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
 d_output = E * slope_output_layer
 Error_at_hidden_layer = d_output.dot(wout.T)
 d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
 wout += hiddenlayer_activations.T.dot(d_output) *lr      
 test=np.sum(d_output, axis=0#,keepdims=True
                ) *lr
 test=test.values   
 bout+= test              
 wh += data_predictors_train.T.dot(d_hiddenlayer) *lr
 test2=np.sum(d_hiddenlayer, axis=0#,keepdims=True
             ) *lr          
 test2=test2.values
 bh += test2





























