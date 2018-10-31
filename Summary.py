# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:23:27 2018

@author: tatsa02
"""

"**************************************************Imports **************************

import os
import numpy as np
import pandas as pd

"*****************************Training data preperation  **************************

os.chdir('C:/Users/tatsa02/Documents/Workspace-Python/ipython-notebooks-master/ipython-notebooks-master/data')
path = os.getcwd() + '/Training_Dataset_full.csv'
data = pd.read_csv(path,header='infer')

#data=data.drop(['INSIDE_OUTSIDE','HIGH_MILESTONE','CUM_AGE_AT_1'], axis=1)
DUMMYS=pd.get_dummies(data,columns=['GEO','ACCOUNT_SEGMENT','INDUSTRY','GBU1','BUSINESS_TYPE','INSIDE_OUTSIDE',
                                    'OPPORTUNITY_SOURCE'],prefix_sep='_')

#Training_data = DUMMYS.loc[DUMMYS.FISCAL_PERIOD.str.contains('2017'),:]
Training_data1=DUMMYS.loc[DUMMYS.FISCAL_PERIOD.isin(pd.Series(['2018 2'])),:]
Training_data2=DUMMYS.loc[DUMMYS.FISCAL_PERIOD.isin(pd.Series(['2018 3 '])),:]
Training_data3=DUMMYS.loc[DUMMYS.FISCAL_PERIOD.isin(pd.Series(['2017 4'])),:]
Training_data4=DUMMYS.loc[DUMMYS.FISCAL_PERIOD.isin(pd.Series(['2018 1'])),:]

    frames = [Training_data1, Training_data2,Training_data3,Training_data4]
    Training_data = pd.concat(frames)
    
"*************************** CAPPING 1  AND 99 th percentile ***********************************

for col in ['AGE_AT_1','AGE_AT_3','AGE_AT_4','AGE_AT_5','CUM_AGE_AT_1','CUM_AGE_AT_3','CUM_AGE_AT_4','CUM_AGE_AT_5']:
    percentiles = data_predictors_train[col].quantile([0.01,0.99]).values
    data_predictors_train[col][data_predictors_train[col] <= percentiles[0]] = percentiles[0]
    data_predictors_train[col][data_predictors_train[col] >= percentiles[1]] = percentiles[1]
    

  "*****************************Finding correlation between   **************************

def cartesian(df1, df2):
    import itertools
    rows = itertools.product(df1.iterrows(), df2.iterrows())

    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    return df.reset_index(drop=True)

def get_Correlated_Pairs(dt_frame,cut_off):
    Correlation_Matrix = dt_frame.corr()
    a = dt_frame.columns.values
    b = dt_frame.columns.values
    a_ = pd.DataFrame({'Prod_1':a})
    b_ = pd.DataFrame({'Prod_2':b})
    Correlation_Mat = cartesian(a_,b_)
    Correlation_Mat = Correlation_Mat[~(Correlation_Mat.Prod_1==Correlation_Mat.Prod_2)]
    Correlation_Mat['Correlation'] = 0
    Correlation_Mat = Correlation_Mat[(Correlation_Mat.Prod_1.isin(Correlation_Matrix.columns.values))]
    Correlation_Mat = Correlation_Mat[(Correlation_Mat.Prod_2.isin(Correlation_Matrix.columns.values))]
    Correlation_Mat.index = range(Correlation_Mat.shape[0])
    for i in range(Correlation_Mat.shape[0]):
        Correlation_Mat.loc[i,'Correlation']= np.abs(Correlation_Matrix.loc[Correlation_Mat.Prod_1[i],Correlation_Mat.Prod_2[i]])
    Correlation_Mat = Correlation_Mat[Correlation_Mat.Correlation>cut_off]
#    print(Correlation_Mat.shape)
    if(Correlation_Mat.shape[0]==0):
        return(Correlation_Mat)
    Correlation_Mat = Correlation_Mat.sort_values(['Correlation'],ascending=False)
    Correlation_Mat['index'] = Correlation_Mat.index
    g = Correlation_Mat.sort_values(['Correlation', 'index']).groupby('Correlation')
    Correlation_Mat['session_number'] = g['index'].rank(method='first', na_option='top')
    import pandasql as ps
    q1 = """ select * from Correlation_Mat Where session_number = 1 """
    Correlation_Mat =  ps.sqldf(q1, locals())
    Correlation_Mat.drop(['index','session_number'],axis=1,inplace=True)
    return(Correlation_Mat)	

get_Correlated_Pairs(data_predictors_train,0.5)

data_predictors_train=data_predictors_train.drop(['AGE_AT_1','AGE_AT_2','AGE_AT_3','AGE_AT_4','AGE_AT_5'], axis=1)

"*****************************Variable importance using Random Forest  **************************

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
le = preprocessing.LabelEncoder()

dropped=data_predictors_train


 for column in dropped :
    if data_predictors_train[column].dtype == type(object):
       # le = LabelEncoder()
        data_predictors_train[column] = le.fit_transform(data_predictors_train[column].astype(str))
        
   
 #data_predictors_train =X ;
 #data_target_train=Y
 
 # Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(data_predictors_train, data_target_train)

# Get numerical feature importances
importances = list(rf.feature_importances_)

feature_list=data_predictors_train.columns
# List of tuples with variable and importance
feature_importances = [(data_predictors_train, round(importance, 2)) 
for data_predictors_train, importance in zip(feature_list, importances)]


# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

"*****************************Removing variables using  importance  **************************

data_predictors_train=data_predictors_train[['INSIDE_OUTSIDE_Outside','INSIDE_OUTSIDE_Inside','CURRENT_MILESTONE','CUM_AGE_AT_5','CUM_AGE_AT_4','TARGETED_CLOSE',
                      'CUM_AGE_AT_2','POSITIVE_HOP_SCORE','CUM_AGE_AT_3','CUM_AGE_AT_1','IS_RENEWAL_ATTACHED']]


"*****************************Model Building **************************

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)


classifier.fit(data_predictors_train, data_target_train)



"*****************************Test data preperation  **************************
Testing_data = DUMMYS.loc[DUMMYS.FISCAL_PERIOD.isin(pd.Series(['2018 1'])),:]
Testing_data = Testing_data.iloc[:,4:]

data_predictors_test=Testing_data.drop(['TARGET'], axis=1)

data_predictors_test=data_predictors_test[['INSIDE_OUTSIDE_Outside','INSIDE_OUTSIDE_Inside','CURRENT_MILESTONE','CUM_AGE_AT_5','CUM_AGE_AT_4','TARGETED_CLOSE',
                      'CUM_AGE_AT_2','POSITIVE_HOP_SCORE','CUM_AGE_AT_3','CUM_AGE_AT_1','IS_RENEWAL_ATTACHED']]



data_target_test=Testing_data['TARGET']

"*********************************************** APPLYING MODEL ON TEST DATA ***********************************

y_pred = classifier.predict(data_predictors_test)

from sklearn.metrics import confusion_matrix

CFM = confusion_matrix(data_target_test, y_pred)


print(CFM)


from sklearn.metrics import recall_score,accuracy_score,precision_score

precision_score(data_target_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred))


y_pred = classifier.predict_proba(data_predictors_test)

y_pred1=[1 if y >0.05 else 0 for y in y_pred[:,1]]

y_pred1=np.asarray(y_pred1)

data_target_test=np.asarray(data_target_test)

CFM1 = confusion_matrix(data_target_test, y_pred1)

print(CFM1)


from sklearn.metrics import recall_score,accuracy_score,precision_score

precision_score(data_target_test,y_pred1)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))

"*****************************Test data preperation2  **************************

Testing_data = DUMMYS.loc[DUMMYS.FISCAL_PERIOD.isin(pd.Series(['2018 2'])),:]
Testing_data = Testing_data.iloc[:,4:]

data_predictors_test=Testing_data.drop(['TARGET'], axis=1)

data_predictors_test=data_predictors_test[['INSIDE_OUTSIDE_Outside','INSIDE_OUTSIDE_Inside','CURRENT_MILESTONE','CUM_AGE_AT_5','CUM_AGE_AT_4','TARGETED_CLOSE',
                      'CUM_AGE_AT_2','POSITIVE_HOP_SCORE','CUM_AGE_AT_3','CUM_AGE_AT_1','IS_RENEWAL_ATTACHED']]



data_target_test=Testing_data['TARGET']

"*********************************************** APPLYING MODEL ON TEST DATA ***********************************

y_pred = classifier.predict(data_predictors_test)

from sklearn.metrics import confusion_matrix

CFM = confusion_matrix(data_target_test, y_pred)


print(CFM)


from sklearn.metrics import recall_score,accuracy_score,precision_score

precision_score(data_target_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred))


y_pred = classifier.predict_proba(data_predictors_test)

y_pred1=[1 if y >0.05 else 0 for y in y_pred[:,1]]

y_pred1=np.asarray(y_pred1)

data_target_test=np.asarray(data_target_test)

CFM1 = confusion_matrix(data_target_test, y_pred1)

print(CFM1)


from sklearn.metrics import recall_score,accuracy_score,precision_score

precision_score(data_target_test,y_pred1)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))


"******************** Ensemble  Gradient Boosting (GBM)********************
#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

#Training_data=Training_data.drop(['CA_ACCOUNT_ID','FISCAL_PERIOD'], axis=1)

Training_data=Training_data[['OPPORTUNITY_NUMBER','INSIDE_OUTSIDE_Outside','INSIDE_OUTSIDE_Inside','CURRENT_MILESTONE','CUM_AGE_AT_5','CUM_AGE_AT_4','TARGETED_CLOSE',
                      'CUM_AGE_AT_2','POSITIVE_HOP_SCORE','CUM_AGE_AT_3','CUM_AGE_AT_1','IS_RENEWAL_ATTACHED','TARGET']]


Testing_data=Testing_data[['OPPORTUNITY_NUMBER','INSIDE_OUTSIDE_Outside','INSIDE_OUTSIDE_Inside','CURRENT_MILESTONE','CUM_AGE_AT_5','CUM_AGE_AT_4','TARGETED_CLOSE',
                      'CUM_AGE_AT_2','POSITIVE_HOP_SCORE','CUM_AGE_AT_3','CUM_AGE_AT_1','IS_RENEWAL_ATTACHED','TARGET']]



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
        
"***************** fitting model*****************

#Choose all predictors except target & IDcols
predictors = [x for x in Training_data.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, Training_data, predictors)


Testing_data[predictors]
y_pred=gbm0.predict(Testing_data[predictors])
from sklearn.metrics import accuracy_score,recall_score

accuracy_score(Testing_data['TARGET'],y_pred)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Testing_data['TARGET'],y_pred)
print(confusion_matrix)
from sklearn.metrics import recall_score,accuracy_score,precision_score

precision_score(Testing_data['TARGET'],y_pred)

accuracy_score(Testing_data['TARGET'],y_pred)

recall_score(Testing_data['TARGET'],y_pred)

print('Accuracy : {:.2f}'.format(gbm0.score(y_pred, Testing_data['TARGET'])))


gbm0.score(y_pred, Testing_data['TARGET'])














