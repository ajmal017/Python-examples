# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:19:21 2018

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
                                                       
#DF_AUTO_2=df[(df['GBU1']=='Automation') & (df['CURRENT_STAGE']==2) ]
#
#
#print (DF_AUTO_2.describe())
#
#df.groupby('GBU1','CURRENT_STAGE').mean()
#
#df.groupby('GBU1','CURRENT_STAGE')['CUMULATIVE_AGE_AT_STAGE1'].mean()



#for GEO in df.GEO.unique():
#   for SEGMENT in df.SEGMENT:
#         for GBU1 in df.GBU1:
#             for CURRENT_STAGE in df.CURRENT_STAGE:
#                 df.groupby(GE0,SEGMENT,GBU1,CURRENT_STAGE)['CUMULATIVE_AGE_AT_STAGE1'].mean()
#                 
#          
#            df_final=pd.DataFrame(data=['GEO','SEGMENT','GBU1','CURRENT_STAGE','SD','MEDIAN','PER_25','PER_50','PER_75','PER_99']) 
#            df_final=df_final.T
#            
            df_final=pd.DataFrame(data=['GEO','SEGMENT','GBU1','CURRENT_STAGE','SD','MEDIAN','MEAN','PER_25','PER_50','PER_75','PER_99','TARGET','COMMENT']) 
            df_final=df_final.T
            
   for GEO in (df.GEO.unique()): 
    for SEGMENT in (df.SEGMENT.unique()):
     for GBU1 in (df.GBU1.unique()):
         for CURRENT_STAGE in (df.CURRENT_STAGE.unique()): 
            for TARGET in (df.TARGET.unique()): 
              if len( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)])==0:
                  continue
              SD=np.std( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_AGE_AT_STAGE1'])
              MEDIAN=np.median( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_AGE_AT_STAGE1'])
              MEAN=np.mean( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_AGE_AT_STAGE1'])
              PER_25= np.percentile( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_AGE_AT_STAGE1'], 25)
              PER_50=np.percentile( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_AGE_AT_STAGE1'], 50)
              PER_75=np.percentile( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_AGE_AT_STAGE1'], 75)
              PER_99=np.percentile( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_AGE_AT_STAGE1'], 99)
              df_temp=pd.DataFrame(data=[GEO,SEGMENT,GBU1,CURRENT_STAGE,SD,MEDIAN,MEAN,PER_25,PER_50,PER_75,PER_99,TARGET,'CUMULATIVE_AGE_AT_STAGE'])
              df_temp=df_temp.T
              #df_temp=df_temp.dropna()
              df_final=df_final.append(df_temp)
              
df_temp.iloc[0:0]

df_temp.drop(df_temp.index, inplace=True)
              
     df_final1=pd.DataFrame(data=['GEO','SEGMENT','GBU1','CURRENT_STAGE','SD','MEDIAN','MEAN','PER_25','PER_50','PER_75','PER_99','TARGET','COMMENT']) 
     df_final1=df_final1.T
     
     
     
              
              
for GEO in (df.GEO.unique()): 
    for SEGMENT in (df.SEGMENT.unique()):
     for GBU1 in (df.GBU1.unique()):
         for CURRENT_STAGE in (df.CURRENT_STAGE.unique()): 
            for TARGET in (df.TARGET.unique()): 
              if len( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)])==0:
                  continue
              SD=np.std( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_OPPORTUNITY_AGE'])
              MEDIAN=np.median( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_OPPORTUNITY_AGE'])
              MEAN=np.mean( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_OPPORTUNITY_AGE'])
              PER_25= np.percentile( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_OPPORTUNITY_AGE'], 25)
              PER_50=np.percentile( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_OPPORTUNITY_AGE'], 50)
              PER_75=np.percentile( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_OPPORTUNITY_AGE'], 75)
              PER_99=np.percentile( df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]['CUMULATIVE_OPPORTUNITY_AGE'], 99)
              df_temp=pd.DataFrame(data=[GEO,SEGMENT,GBU1,CURRENT_STAGE,SD,MEDIAN,MEAN,PER_25,PER_50,PER_75,PER_99,TARGET,'CUMULATIVE_OPPORTUNITY_AGE'])
              df_temp=df_temp.T
              #df_temp=df_temp.dropna()
              df_final1=df_final1.append(df_temp)
              
              
    df[(df['GEO']==GEO) & (df['SEGMENT']==SEGMENT) & (df['GBU1']==GBU1) & (df['CURRENT_STAGE']==CURRENT_STAGE) &  (df['TARGET']==TARGET)]
#              
#              
#   df_final=df_final.drop_duplicates()
              
           
#              
#import numpy
#import scipy
#
#stddev = 0
#average = 0
#scores = [0,1,2,3,4]
#average = numpy.mean(scores) #or calculate it yourself
#diffsquared = 0
#sum_diffsquared = 0
#for val in scores:
#  diffsquared = (val-average)**2
#  sum_diffsquared = diffsquared + sum_diffsquared
#stddev = ((sum_diffsquared)/len(scores))**(1/2) # corrected
#print(stddev)
#print(numpy.sqrt(numpy.var(scores))
               
              
                   

#                                                       
#                                                       
#                                                       
#df.groupby(['INSIDE_OUTSIDE','TARGET'])['TARGET'].count()
#
#df.groupby(['GEO','TARGET'])['TARGET'].count()

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

        
       
#X_train, X_test = train_test_split(df, test_size=0.2,random_state=42)
        
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
        
"***************** fitting model*****************
#Choose all predictors except target & IDcols
predictors = [x for x in X_train1.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)

gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=9, min_samples_split=1200,
                                         min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)


X_train1['TARGET'] = X_train1.TARGET.astype(int)

#cv_score = cross_validation.cross_val_score(gbm0, X_train[predictors], X_train['TARGET'], cv=5, scoring='roc_auc')
    


#gbm0.fit(X_train[predictors], X_train['TARGET'])

modelfit(gbm0, X_train1,predictors)


modelfit(gbm_tuned_1, X_train1,predictors)

    
#Testing_data[predictors]
y_pred=gbm0.predict(X_test1[predictors])

y_pred=gbm0.predict_proba(X_test1[predictors])

y_pred=gbm_tuned_1.predict(X_test1[predictors])

y_pred=gbm_tuned_1.predict_proba(X_test1[predictors])


y_pred1=[1 if y >0.6 else 0 for y in y_pred[:,1]]

#y_pred1=gbm0.predict_proba(X_test[predictors])





X_test1['TARGET'] = X_test1.TARGET.astype(int)

#y_pred2=[1 if y >0.2 else 0 for y in y_pred1[:,1]]

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(X_test1['TARGET'],y_pred)
print(confusion_matrix)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(X_test1['TARGET'],y_pred1)
print(confusion_matrix)

#from sklearn.metrics import confusion_matrix
#confusion_matrix = confusion_matrix(X_test['TARGET'],y_pred2)
#print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(X_test1['TARGET'],y_pred))

#from sklearn.metrics import classification_report
#print(classification_report(X_test['TARGET'],y_pred2))


from sklearn.metrics import recall_score,accuracy_score,precision_score



accuracy_score(X_test1['TARGET'],y_pred)

X_test2['Y_pred']=y_pred

 X_test2.to_csv('X_test2.csv')

X_test3=X_test2[(X_test2['TARGET']==1) & (X_test2['Y_pred']==0) ]


X_test2.groupby(['GEO','TARGET'])['TARGET'].count()


X_test2['TARGET']


#*******************Grid search************************#

#Choose all predictors except target & IDcols
predictors = [x for x in X_train1.columns if x not in [target, IDcol]]
param_test1 = {'n_estimators':np.arange(20,81,10)}
gsearch1 = GridSearchCV(estimator = 
                        GradientBoostingClassifier(learning_rate=0.1, 
                                                               min_samples_split=500,
                                                               min_samples_leaf=50,
                                                               max_depth=8,
                                                               max_features='sqrt',
                                                               subsample=0.8,
                                                               random_state=10), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
                        
X_train1['TARGET'] = X_train1.TARGET.astype(int)

if __name__ == '__main__':

                       
   gsearch1.fit(X_train1[predictors],X_train1[target])



























              



