# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 18:48:18 2018

@author: tatsa02
"""

import pyodbc 
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4



cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                        "Server=USILDB55;"
                        "Database=Win_Loss;"
                        "uid=a_user;pwd=Welcome@1")
df = pd.read_sql_query('''SELECT *,
     CASE WHEN STAGE_METRIC<=1 THEN 1  ELSE 0 END AS 'STAGE <=1 SIGMA',
	 CASE WHEN STAGE_METRIC>1 AND STAGE_METRIC<=2 THEN 1  ELSE 0 END AS 'STAGE >1and<=2 SIGMA',
	 CASE WHEN STAGE_METRIC>2 THEN 1  ELSE 0 END AS 'STAGE >2 SIGMA',
	 CASE WHEN OPP_AGE_METRIC<=1 THEN 1  ELSE 0 END AS 'OPP <=1 SIGMA',
	 CASE WHEN OPP_AGE_METRIC>1 AND OPP_AGE_METRIC<=2 THEN 1  ELSE 0 END AS 'OPP >1and<=2 SIGMA',
	 CASE WHEN OPP_AGE_METRIC>2 THEN 1  ELSE 0 END AS 'OPP >2 SIGMA'

FROM
(SELECT 
GEO,
OPPORTUNITY_NUMBER,
CA_ACCOUNT_ID,
SEGMENT,
OPPORTUNITY_SOURCE,
GBU1,
INSIDE_SALES,
INSIDE_OUTSIDE,
RTM,
IS_RENEWAL_ATTACHED,
IS_EDUCATION_ATTACHED,
IS_SERVICES_ATTACHED,
IS_CAPACITY_ATTACHED,
IS_UPGRADE_ATTACHED,
CURRENT_STAGE,
CUMULATIVE_AGE_AT_STAGE1,
CUMULATIVE_OPPORTUNITY_AGE,
NEGATIVE_HOP_SCORE,
POSITIVE_HOP_SCORE,
CREATED_QUARTER,
CLOSED_QUARTER,
TARGET,
SUM(STAGE_METRIC) AS STAGE_METRIC,
SUM(OPP_AGE_METRIC) AS OPP_AGE_METRIC
FROM(

SELECT a.*,b.CREATED_QUARTER,b.CLOSED_QUARTER,b.TARGET,
stat.COMMENT,STAT.MEAN,STAT.SD,
CASE WHEN STAT.COMMENT='CUMULATIVE_AGE_AT_STAGE' THEN ((CAST(A.CUMULATIVE_AGE_AT_STAGE1 AS FLOAT)-STAT.MEAN)/STAT.SD) ELSE 0 END 
AS STAGE_METRIC,
CASE WHEN STAT.COMMENT='CUMULATIVE_OPPORTUNITY_AGE' THEN ((CAST(A.CUMULATIVE_OPPORTUNITY_AGE AS FLOAT)-STAT.MEAN)/STAT.SD) ELSE 0 END 
AS OPP_AGE_METRIC
 FROM HEALTHSCORE_NEW_MODEL_FINAL_DATA_V2  a
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
                              MONTH(MAX(CAST(NEXT_QUARTER_STARTER_DAY AS DATE))) closed_month  FROM [dbo].[HEALTHSCORE_NEW_MODEL_FINAL_DATA_ASHEESH]
                                           GROUP BY OPPORTUNITY_NUMBER,TARGET
   ) xx
   )b
   on a.OPPORTUNITY_NUMBER=b.OPPORTUNITY_NUMBER
   AND a.CUMULATIVE_OPPORTUNITY_AGE>30
   join STATS_HS stat
   on A.GEO=stat.GEO
 AND A.SEGMENT=stat.SEGMENT
 AND A.GBU1=stat.GBU1
 AND A.CURRENT_STAGE=stat.CURRENT_STAGE
 AND STAT.TARGET=1
 AND STAT.SD<>0
 )XX
 GROUP BY 
 GEO,
OPPORTUNITY_NUMBER,
CA_ACCOUNT_ID,
SEGMENT,
OPPORTUNITY_SOURCE,
GBU1,
INSIDE_SALES,
INSIDE_OUTSIDE,
RTM,
IS_RENEWAL_ATTACHED,
IS_EDUCATION_ATTACHED,
IS_SERVICES_ATTACHED,
IS_CAPACITY_ATTACHED,
IS_UPGRADE_ATTACHED,
CURRENT_STAGE,
CUMULATIVE_AGE_AT_STAGE1,
CUMULATIVE_OPPORTUNITY_AGE,
NEGATIVE_HOP_SCORE,
POSITIVE_HOP_SCORE,
CREATED_QUARTER,
CLOSED_QUARTER,
TARGET)YY

 --SELECT * FROM STATS_HS WHERE GEO='APJ' AND SEGMENT='Commercial' AND GBU1='Agile Management'
   ''', cnxn)
                                                       
X_train1, X_test1 = train_test_split(df, test_size=0.2,random_state=42)

X_train2=X_train1.copy()
X_test2=X_test1.copy()
                                                       
 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
 
       
        
for column in ['GEO', 'SEGMENT','OPPORTUNITY_SOURCE','GBU1','INSIDE_SALES', 'INSIDE_OUTSIDE', 'RTM']:
    if X_train1[column].dtypes == type(object):
       # le = LabelEncoder()
        X_train1[column] = le.fit_transform(X_train1[column].astype(str))
        
X_train1=X_train1.drop(['CA_ACCOUNT_ID','INSIDE_OUTSIDE'], axis=1)

#df.groupby(['INSIDE_OUTSIDE','TARGET'])['TARGET'].count()


for column in ['GEO', 'SEGMENT','OPPORTUNITY_SOURCE','GBU1','INSIDE_SALES', 'INSIDE_OUTSIDE', 'RTM']:
    if X_test1[column].dtypes == type(object):
       # le = LabelEncoder()
        X_test1[column] = le.fit_transform(X_test1[column].astype(str))
        
        X_test1=X_test1.drop(['CA_ACCOUNT_ID','INSIDE_OUTSIDE'], axis=1)

X_train1_1=X_train1.drop(['TARGET'], axis=1)

y_train1_1=X_train1['TARGET']

# Train the model on training data
rf.fit(X_train1_1, y_train1_1)


X_test1_1=X_test1.drop(['TARGET'], axis=1)

y_test1_1=X_test1['TARGET']

def train_using_gini(X_train, X_test, y_train):
 
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
    
#     from sklearn import preprocessing
#     le = preprocessing.LabelEncoder()
#     
#     X_train=pd.DataFrame(X_train)
#    
#    for column_name in X_train.columns:
#       
#        if X_train[column_name].dtype == object:
#           X_train[column_name] = le.fit_transform(X_train[column_name])
#        else:
#            pass
 
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini
     
# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
 
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
 
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
 
 
# Function to make predictions
def prediction(X_test, clf_object):
 
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
     
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
     
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
     
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
     
    print("Report : ",
    classification_report(y_test, y_pred))
 
# Driver code
def main():
     
    # Building Phase
   # data = importdata()
    #X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train1_1, X_test1_1, y_train1_1)
    clf_entropy = tarin_using_entropy(X_train1_1, X_test1_1, y_train1_1)
     
    # Operational Phase
    print("Results Using Gini Index:")
     
    # Prediction using gini
    y_pred_gini = prediction(X_test1_1, clf_gini)
    cal_accuracy(y_test1_1, y_pred_gini)
     
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test1_1, clf_entropy)
    cal_accuracy(y_test1_1, y_pred_entropy)
     
     
# Calling main function
if __name__=="__main__":
    main()
    
    
#Import Library
from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVC() 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X_train1_1, y_train1_1)
#model.score(X_train, y_train)
#Predict Output
predicted= model.predict(X_test1_1)


from sklearn.metrics import confusion_matrix
# Confusion Matrix
print(" Confusion matrix ", confusion_matrix(y_train1_1, rf.predict(X_train1_1)))

accuracy_score(y_train1_1,rf.predict(X_train1_1))

from sklearn.metrics import classification_report
print(classification_report(y_test1_1,rf.predict(X_test1_1)))
