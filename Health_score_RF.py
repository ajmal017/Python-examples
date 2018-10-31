# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:51:55 2018

@author: tatsa02
"""
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
import pyodbc 
import numpy as np
import pandas as pd

cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                        "Server=USILDB55;"
                        "Database=Win_Loss;"
                        "uid=a_user;pwd=Welcome@1")
balance_data = pd.read_sql_query('''SELECT * FROM VW_HEALTHSCORE_NEW_MODEL_FINAL_DATA_V1  a
   join(
   select OPPORTUNITY_NUMBER,TARGET,
          case when created_month in (2,3,4) then 4
               when created_month in (5,6,7) then 1
                       when created_month in (8,9,10) then 2
                       when created_month in (1,11,12) then 3 end as CREATED_QUARTER,
               case 
                    when closed_month in (2,3,4) then 4
               when closed_month in (5,6,7) then 1
                       when closed_month in (8,9,10) then 2
                       when closed_month in (1,11,12) then 3 end as CLOSED_QUARTER
                                                       from(

   SELECT OPPORTUNITY_NUMBER,TARGET, MONTH(MIN(CAST(NEXT_QUARTER_STARTER_DAY AS DATE))) created_month ,
                              MONTH(MAX(CAST(NEXT_QUARTER_STARTER_DAY AS DATE))) closed_month  FROM [dbo].[HEALTHSCORE_NEW_MODEL_FINAL_DATA]
                                           GROUP BY OPPORTUNITY_NUMBER,TARGET
   ) xx
   )b
   on a.OPPORTUNITY_NUMBER=b.OPPORTUNITY_NUMBER''', cnxn)
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
    if balance_data[column].dtypes == type(object):
       # le = LabelEncoder()
        balance_data[column] = le.fit_transform(balance_data[column].astype(str))

X = balance_data.iloc[:,1:31]
Y = balance_data.iloc[:,32]
 
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
     

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)


# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
errors=pd.DataFrame(errors)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')




X_train.columns

# Get numerical feature importances
importances = list(rf.feature_importances_)

feature_list=X_train.columns
# List of tuples with variable and importance
feature_importances = [(X_train, round(importance, 2)) 
for X_train, importance in zip(feature_list, importances)]


# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)



from sklearn.metrics import accuracy_score,recall_score,precision_score

accuracy_score(y_test,rf.predict(X_test))

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')



from sklearn.metrics import confusion_matrix
# Confusion Matrix
print(" Confusion matrix ", confusion_matrix(Train_TargetVar, predictions))

from sklearn.metrics import confusion_matrix
# Confusion Matrix
print(" Confusion matrix ", confusion_matrix(Train_TargetVar, predictions))


















