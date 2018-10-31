# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:42:56 2018

@author: tatsa02
"""
import os
#import numpy as np
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

corr=data.corr()

import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pandas import pandasql

import pandasql

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
    Correlation_Mat = Correlation_Mat.sort_values(['Correlation'],ascending=False)
    Correlation_Mat['index'] = Correlation_Mat.index
    g = Correlation_Mat.sort_values(['Correlation', 'index']).groupby('Correlation')
    Correlation_Mat['session_number'] = g['index'].rank(method='first', na_option='top')
    import pandasql as ps
    q1 = """ select * from Correlation_Mat Where session_number = 1 """
    Correlation_Mat =  ps.sqldf(q1, locals())
    Correlation_Mat.drop(['index','session_number'],axis=1,inplace=True)
    return(Correlation_Mat)   

data_1=data.iloc[:, 8:99]

data_1=  data_1.drop(['OPPORTUNITY_NUMBER','AGE_AT_3','CUM_AGE_AT_3','AGE_AT_1'], axis=1)
X_ = data_1.drop(['TARGET'],axis=1)
Y_ = data_1['TARGET']
features = "+".join(X_.columns)         

Corr_Pairs = get_Correlated_Pairs(X_,0.6)

#%capture
#gather features



# get y and X dataframes based on this regression:
y, X = dmatrices('TARGET~' + features, data_1, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

#-----------
data_1=data.iloc[:, 8:99]

data_1=  data_1.drop(['AGE_AT_3','CUM_AGE_AT_3','CUM_AGE_AT_2','AGE_AT_1','INDUSTRY_?','OPPORTUNITY_NUMBER'], 
                     axis=1)

df=data_1.columns.tolist()

df=pd.DataFrame({'col':df})

df=pd.DataFrame(df)

data_1.columns=['AGE_AT_2',
'AGE_AT_4',
'AGE_AT_6',
'AGE_AT_8',
'AGE_AT_9',
'CUM_AGE_AT_1',
'CUM_AGE_AT_4',
'CUM_AGE_AT_6',
'CUM_AGE_AT_8',
'CUM_AGE_AT_9',
'TARGETED_CLOSE',
'HIGH_MILESTONE',
'POSITIVE_HOP_SCORE',
'NEGATIVE_HOP_SCORE',
'IS_RENEWAL_ATTACHED',
'IS_EDUCATION_ATTACHED',
'IS_Services_ATTACHED',
'SEGMENT_GROWTH',
'IS_ABSOLUTE_NEWLOGO',
'IS_NEWLOGO_GBU',
'TARGET',
#'OPPORTUNITY_NUMBER',
'AGE_AT_2',
'AGE_AT_4',
'AGE_AT_6',
'AGE_AT_8',
'AGE_AT_9',
'CUM_AGE_AT_1',
'CUM_AGE_AT_4',
'CUM_AGE_AT_6',
'CUM_AGE_AT_8',
'CUM_AGE_AT_9',
'TARGETED_CLOSE',
'HIGH_MILESTONE',
'POSITIVE_HOP_SCORE',
'NEGATIVE_HOP_SCORE',
'IS_RENEWAL_ATTACHED',
'IS_EDUCATION_ATTACHED',
'IS_Services_ATTACHED',
'SEGMENT_GROWTH',
'IS_ABSOLUTE_NEWLOGO',
'IS_NEWLOGO_GBU',
'TARGET',
'GEO_APJ',
'GEO_EMEA',
'GEO_LA',
'GEO_PS_CAN',
'ACCOUNT_SEGMENT_Commercial',
'ACCOUNT_SEGMENT_GOLD',
'ACCOUNT_SEGMENT_PLATINUM',
'ACCOUNT_SEGMENT_SILVER',
'INDUSTRY_Construction',
'INDUSTRY_Finance_Insurance_And_Real_Estate',
'INDUSTRY_Manufacturing',
'INDUSTRY_Mining',
'INDUSTRY_Nonclassifiable_Establishments',
'INDUSTRY_Public_Administration',
'INDUSTRY_Retail_Trade',
'INDUSTRY_Services',
'INDUSTRY_Transportation_Communications_And_Utilities',
'INDUSTRY_Unknown_SIC_Division',
'INDUSTRY_Wholesale_Trade',
'GBU1_Agile_Management',
'GBU1_Agile_Operations',
'GBU1_Automation',
'GBU1_Continuous_Delivery',
'GBU1_Developer_Products',
'GBU1_Mainframe',
'GBU1_Other',
'GBU1_Security',
'INSIDE_OUTSIDE_Inside',
'INSIDE_OUTSIDE_Outside',
'BUSINESS_TYPE_',
'BUSINESS_TYPE_Capacity',
'BUSINESS_TYPE_MF_Capacity',
'BUSINESS_TYPE_Maintenance',
'BUSINESS_TYPE_New',
'BUSINESS_TYPE_Upgrade',
'OPPORTUNITY_SOURCE_DIRECT',
'OPPORTUNITY_SOURCE_ISR_OUTBOUND',
'OPPORTUNITY_SOURCE_MKT_ISR',
'OPPORTUNITY_SOURCE_PARTNER'
]


   
X_ = data_1.drop(['TARGET',],axis=1)
Y_ = data_1['TARGET']
features = "+".join(X_.columns)


# get y and X dataframes based on this regression:
y, X = dmatrices('TARGET~' + features, data_1, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

vif.round(1)



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

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)


classifier.fit(data_predictors_train, data_target_train)

X_test_1=X_test.iloc[:, 8:99]

#data1['TARGET']

data_predictors_test=X_test_1.drop(['TARGET','OPPORTUNITY_NUMBER'], axis=1)

data_target_test=X_test_1['TARGET']

data_target_test=data_target_test.iloc[:,1]

data_target_test=pd.DataFrame(data_target_test)


y_pred = classifier.predict(data_predictors_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(data_target_test, y_pred)
print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.
      format(classifier.score(data_predictors_test, data_target_test)))

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred))




























