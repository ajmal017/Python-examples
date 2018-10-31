# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:03:50 2018

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

########## XGB #################################

target = 'TARGET'
IDcol = 'OPPORTUNITY_NUMBER'

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['TARGET'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['TARGET'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['TARGET'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['TARGET'], dtrain_predprob))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        
        
        #Choose all predictors except target & IDcols
predictors = [x for x in X_train1.columns if x not in [target, IDcol]]

X_train1['TARGET'] = X_train1.TARGET.astype(int)

param_test1 = {'n_estimators':np.arange(20,81,10)}
gsearch1 = GridSearchCV(estimator = 
                        GradientBoostingClassifier(learning_rate=0.1, 
                                                               min_samples_split=500,
                                                               min_samples_leaf=50,
                                                               max_depth=8,
                                                               max_features='sqrt',
                                                               subsample=0.8,
                                                               random_state=10), 
param_grid = param_test1, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
                        
                        
param_test2 = {'max_depth':np.arange(5,16,2), 'min_samples_split':np.arange(200,1001,200)}

gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, 
                                                                n_estimators=60, 
                                                                max_features='sqrt', 
                                                                subsample=0.8, 
                                                                random_state=10), 
         param_grid = param_test2, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
                        
                        
param_test2b = {'min_child_weight':[6,8,10,12]}

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
                        
 gsearch2b.fit(train[predictors],train[target])


                        
                        
param_test3 = {'min_samples_split':np.arange(1000,2100,200), 'min_samples_leaf':np.arange(30,71,10)}

gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, 
                                                               n_estimators=60,
                                                               max_depth=9,
                                                               max_features='sqrt', 
                                                               subsample=0.8, 
                                                               random_state=10), 
param_grid = param_test3, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
                        
                    

                        
X_train1['TARGET'] = X_train1.TARGET.astype(int)

gsearch1.fit(X_train1[predictors],X_train1[target])

gsearch2.fit(X_train1[predictors],X_train1[target])

gsearch3.fit(X_train1[predictors],X_train1[target])

y_pred=gsearch1.predict(X_test1[predictors])

y_pred=gsearch2.predict(X_test1[predictors])

y_pred=gsearch3.predict(X_test1[predictors])

y_pred=gsearch2.predict(X_test1[predictors])

y_pred=gsearch1.predict_proba(X_test1[predictors])

X_test1['TARGET'] = X_test1.TARGET.astype(int)

#y_pred2=[1 if y >0.2 else 0 for y in y_pred1[:,1]]

    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(X_test1['TARGET'],y_pred)
    print(confusion_matrix)


from sklearn.metrics import classification_report
print(classification_report(X_test1['TARGET'],y_pred))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(X_test1['TARGET'],y_pred1)
print(confusion_matrix)

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)

X_train1_1=X_train1.drop(['TARGET'], axis=1)

y_train1_1=X_train1['TARGET']

# Train the model on training data
rf.fit(X_train1_1, y_train1_1)


X_test1_1=X_test1.drop(['TARGET'], axis=1)

y_test1_1=X_test1['TARGET']

predictions = rf.predict(X_test1_1)

from sklearn.metrics import accuracy_score,recall_score,precision_score

accuracy_score(y_test1_1,rf.predict(X_test1_1))


from sklearn.metrics import confusion_matrix
# Confusion Matrix
print(" Confusion matrix ", confusion_matrix(y_train1_1, rf.predict(X_train1_1)))

accuracy_score(y_train1_1,rf.predict(X_train1_1))

from sklearn.metrics import classification_report
print(classification_report(y_test1_1,rf.predict(X_test1_1)))


from sklearn.metrics import confusion_matrix
# Confusion Matrix
print(" Confusion matrix ", confusion_matrix(y_test1_1, predictions))

importances = list(rf.feature_importances_)

feature_list=X_train1_1.columns
# List of tuples with variable and importance
feature_importances = [(X_train1_1, round(importance, 2)) 
for X_train1_1, importance in zip(feature_list, importances)]


# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)


classifier.fit(X_train1_1, y_train1_1)

y_pred = classifier.predict(X_test1_1)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test1_1, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test1_1,classifier.predict(X_test1_1)))



                        
                        