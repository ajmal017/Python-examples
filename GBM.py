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

data=  data[['FISCAL_PERIOD','OPPORTUNITY_NUMBER','CA_ACCOUNT_ID','INDUSTRY','GEO','ACCOUNT_SEGMENT','GBU1','INSIDE_OUTSIDE','BUSINESS_TYPE',
             'OPPORTUNITY_SOURCE','AGE_AT_2','CUM_AGE_AT_2','TARGET']]

#data=data.drop(['INSIDE_OUTSIDE','HIGH_MILESTONE','CUM_AGE_AT_1'], axis=1)
DUMMYS=pd.get_dummies(data,columns=['GEO','ACCOUNT_SEGMENT','INDUSTRY','GBU1','BUSINESS_TYPE','INSIDE_OUTSIDE',
                                    'OPPORTUNITY_SOURCE'],prefix_sep='_')

Training_data = DUMMYS.loc[DUMMYS.FISCAL_PERIOD.str.contains('2017'),:]
Training_data1=DUMMYS.loc[DUMMYS.FISCAL_PERIOD.isin(pd.Series(['2018 3'])),:]
Training_data2=DUMMYS.loc[DUMMYS.FISCAL_PERIOD.isin(pd.Series(['2017 4'])),:]
Training_data3=DUMMYS.loc[DUMMYS.FISCAL_PERIOD.isin(pd.Series(['2018 1'])),:]
Training_data4=DUMMYS.loc[DUMMYS.FISCAL_PERIOD.isin(pd.Series(['2018 2'])),:]

    frames = [Training_data1, Training_data2,Training_data3,Training_data4]
    Training_data = pd.concat(frames)
    
    Training_data.drop_duplicates()

"*****************************Testing data preperation  **************************
    
os.chdir('C:/Users/tatsa02/Documents/Workspace-Python/ipython-notebooks-master/ipython-notebooks-master/data')
path = os.getcwd() + '/Testing_Dataset_2017-10-01.csv'
data = pd.read_csv(path,header='infer')

data=  data[['FISCAL_PERIOD','OPPORTUNITY_NUMBER','CA_ACCOUNT_ID','INDUSTRY','GEO','ACCOUNT_SEGMENT','GBU1','INSIDE_OUTSIDE','BUSINESS_TYPE',
             'OPPORTUNITY_SOURCE','AGE_AT_2','CUM_AGE_AT_2','TARGET']]
DUMMYS1=pd.get_dummies(data,columns=['GEO','ACCOUNT_SEGMENT','INDUSTRY','GBU1','BUSINESS_TYPE','INSIDE_OUTSIDE',
                                    'OPPORTUNITY_SOURCE'],prefix_sep='_')
Testing_data = DUMMYS1.loc[DUMMYS1.FISCAL_PERIOD.isin(pd.Series(['2018 4'])),:]



"*************************** CAPPING 1  AND 99 th percentile ***********************************

for col in ['AGE_AT_1','AGE_AT_3','AGE_AT_4','AGE_AT_5','CUM_AGE_AT_1','CUM_AGE_AT_3','CUM_AGE_AT_4','CUM_AGE_AT_5']:
    
#for col in ['AGE_AT_2','CUM_AGE_AT_2']:
    percentiles = Training_data[col].quantile([0.01,0.99]).values
    Training_data[col][Training_data[col] <= percentiles[0]] = percentiles[0]
    Training_data[col][Training_data[col] >= percentiles[1]] = percentiles[1]
    
    Training_data=Training_data.drop(['CA_ACCOUNT_ID','FISCAL_PERIOD'], axis=1)
    
"*************************** CAPPING 1  AND 99 th percentile ***********************************

for col in ['AGE_AT_1','AGE_AT_3','AGE_AT_4','AGE_AT_5','CUM_AGE_AT_1','CUM_AGE_AT_3','CUM_AGE_AT_4','CUM_AGE_AT_5']:
    percentiles = Testing_data[col].quantile([0.01,0.99]).values
    Testing_data[col][Testing_data[col] <= percentiles[0]] = percentiles[0]
    Testing_data[col][Testing_data[col] >= percentiles[1]] = percentiles[1]
    
    
    
    Testing_data=Testing_data.drop(['INSIDE_OUTSIDE_LIKELY'], axis=1)
    Testing_data=Testing_data.drop(['CA_ACCOUNT_ID','FISCAL_PERIOD'], axis=1)
    
    Testing_data=Testing_data[Testing_data.AGE_AT_2>0]
    
    Testing_data=Testing_data[Testing_data.CURRENT_MILESTONE==2]
    
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

"******************selecting the important features *************************

#Training_data=Training_data.drop(['CA_ACCOUNT_ID','FISCAL_PERIOD'], axis=1)

Training_data=Training_data[['OPPORTUNITY_NUMBER','INSIDE_OUTSIDE_Outside','INSIDE_OUTSIDE_Inside','CURRENT_MILESTONE','CUM_AGE_AT_5','CUM_AGE_AT_4','TARGETED_CLOSE',
                      'CUM_AGE_AT_2','POSITIVE_HOP_SCORE','CUM_AGE_AT_3','CUM_AGE_AT_1','IS_RENEWAL_ATTACHED','TARGET']]


Testing_data=Testing_data[['OPPORTUNITY_NUMBER','INSIDE_OUTSIDE_Outside','INSIDE_OUTSIDE_Inside','CURRENT_MILESTONE','CUM_AGE_AT_5','CUM_AGE_AT_4','TARGETED_CLOSE',
                      'CUM_AGE_AT_2','POSITIVE_HOP_SCORE','CUM_AGE_AT_3','CUM_AGE_AT_1','IS_RENEWAL_ATTACHED','TARGET']]


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
        
Cor_pairs = get_Correlated_Pairs(Training_data,0.5)

"***************** fitting model*****************

#Choose all predictors except target & IDcols
predictors = [x for x in Training_data.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, Training_data, predictors)
#Testing_data[predictors]
y_pred=gbm0.predict(Testing_data[predictors])

proba=gbm0.predict_proba(Testing_data[predictors])

[1 if y >0.14 else 0 for y in proba[:,1]]

y_pred1=[1 if y >0.01 else 0 for y in proba[:,1]]



df1=pd.DataFrame(proba)
df2=pd.DataFrame(Testing_data['TARGET'])
frames=[df1, df2]
pd.concat(frames)



from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Testing_data['TARGET'],y_pred)
print(confusion_matrix)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Testing_data['TARGET'],y_pred1)
print(confusion_matrix)
#from sklearn.metrics import recall_score,accuracy_score,precision_score
#
#precision_score(Testing_data['TARGET'],y_pred)
#
#accuracy_score(Testing_data['TARGET'],y_pred)
#
#recall_score(Testing_data['TARGET'],y_pred)

from sklearn.metrics import classification_report
print(classification_report(Testing_data['TARGET'],y_pred))

from sklearn.metrics import classification_report
print(classification_report(Testing_data['TARGET'],y_pred1))

"*****************************GBU TUNED 1******************************

predictors = [x for x in Training_data.columns if x not in [target, IDcol]]
gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=9, min_samples_split=1200,
                                         min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
modelfit(gbm_tuned_1, Training_data, predictors)
y_pred=gbm_tuned_1.predict(Testing_data[predictors])

from sklearn.metrics import accuracy_score,recall_score

accuracy_score(Testing_data['TARGET'],gbm_tuned_1.predict(Testing_data[predictors]))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Testing_data['TARGET'], gbm_tuned_1.predict(Testing_data[predictors]))
confusion_matrix

PROBA1=gbm_tuned_1.predict_proba(Testing_data[predictors])

proba1=PROBA1
