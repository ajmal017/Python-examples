# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.
 
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os
 
# Function importing Dataset
#def importdata():
#    balance_data = pd.read_csv(
#'https://archive.ics.uci.edu/ml/machine-learning-'+
#'databases/balance-scale/balance-scale.data',
#    sep= ',', header = None)
    
 os.chdir('C:/Users/tatsa02/Documents/Workspace-Python/ipython-notebooks-master/ipython-notebooks-master/data')
 #print (os.getcwd())
 path = os.getcwd() + '/HELATHSCORE_DATA.csv'
 balance_data = pd.read_csv(path,header='infer')
 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
 
 for column in balance_data.columns:
    if balance_data[column].dtype == type(object):
       # le = LabelEncoder()
        balance_data[column] = le.fit_transform(balance_data[column].astype(str))

X = balance_data.iloc[:,1:31]
Y = balance_data.iloc[:,32]
 
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 42)


#Import Library
from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVC() 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X_train, y_train)
model.score(X_train, y_train)
#Predict Output
predicted= model.predict(X_test)


from sklearn.metrics import accuracy_score,recall_score,precision_score

accuracy_score(y_test,model.predict(X_test))



