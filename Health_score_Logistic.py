"""
Created on Mon Jun 11 12:47:32 2018

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
DUMMYS=DUMMYS[DUMMYS.AGE_AT_2>30]

DUMMYS=DUMMYS.drop(['AGE_AT_1','AGE_AT_3','AGE_AT_4','AGE_AT_5','CUM_AGE_AT_1','CUM_AGE_AT_3','CUM_AGE_AT_4','CUM_AGE_AT_5'], axis=1)

#help(pd.get_dummies)
#DUMMY
Training_data = DUMMYS.loc[DUMMYS.FISCAL_PERIOD.str.contains('2017'),:]

Training_data = Training_data.iloc[:,3:55]

#Corr_Pairs = get_Correlated_Pairs(Training_data.drop(['TARGET'], axis=1),0.5)

data_predictors_train=Training_data.drop(['TARGET'], axis=1)

data_target_train=Training_data['TARGET']

# basic plot
plt.boxplot(data_predictors_train)

"*************************** CAPPING 1  AND 99 th percentile ***********************************

for col in ['AGE_AT_1','AGE_AT_3','AGE_AT_4','AGE_AT_5','CUM_AGE_AT_1','CUM_AGE_AT_3','CUM_AGE_AT_4','CUM_AGE_AT_5']:
    percentiles = data_predictors_train[col].quantile([0.01,0.99]).values
    data_predictors_train[col][data_predictors_train[col] <= percentiles[0]] = percentiles[0]
    data_predictors_train[col][data_predictors_train[col] >= percentiles[1]] = percentiles[1]
    
    corr_1=get_Correlated_Pairs(data_predictors_train,0.3)



"*****************************Model Building **************************

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)


classifier.fit(data_predictors_train, data_target_train)

"*****************************Test data preperation  **************************


os.chdir('C:/Users/tatsa02/Documents/Workspace-Python/ipython-notebooks-master/ipython-notebooks-master/data')
path = os.getcwd() + '/Testing_Dataset_2017-04-01.csv'
data = pd.read_csv(path,header='infer')

#data=data.drop(['INSIDE_OUTSIDE','HIGH_MILESTONE','CUM_AGE_AT_1'], axis=1)
DUMMYS1=pd.get_dummies(data,columns=['GEO','ACCOUNT_SEGMENT','INDUSTRY','GBU1','BUSINESS_TYPE','INSIDE_OUTSIDE',
                                    'OPPORTUNITY_SOURCE'],prefix_sep='_')

DUMMYS1=DUMMYS1[DUMMYS1.AGE_AT_2>30]

DUMMYS1=DUMMYS1.drop(['AGE_AT_1','AGE_AT_3','AGE_AT_4','AGE_AT_5','CUM_AGE_AT_1','CUM_AGE_AT_3','CUM_AGE_AT_4','CUM_AGE_AT_5'], axis=1)



Testing_data = DUMMYS1.loc[DUMMYS1.FISCAL_PERIOD.isin(pd.Series(['2018 1'])),:]
Testing_data = Testing_data.iloc[:,3:56]

Testing_data=Testing_data.drop(['GBU1_Security',], axis=1)

data_predictors_test=Testing_data.drop(['TARGET'], axis=1)

data_target_test=Testing_data['TARGET']

import matplotlib.pyplot  as plt



"*****************************Variable importance using Random Forest  **************************

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
le = preprocessing.LabelEncoder()

dropped=data.columns.drop('FISCAL_PERIOD')
 
 for column in dropped :
    if data[column].dtype == type(object):
       # le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        
data=data.loc[data.FISCAL_PERIOD.str.contains('2017'),:]

X = data.iloc[:,3:33]

Y = pd.DataFrame(data=data, columns=['TARGET'])
 
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split
    (  X, Y, test_size = 0.3, random_state = 42)
    
 X_train =X ;
 y_train=Y
 
 # Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)

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


 "*****************************Helper Functions  **************************
 
 
 def get_Outliers(dt_Frame):
    Cols = dt_Frame.isnull().any()
    Null_Cols = list(Cols[Cols==True].index)
    get_Cols = []
    cols_Remove = []
    for i in range(len(Null_Cols)):
        Num_Nulls = dt_Frame[dt_Frame[[Null_Cols[i]]].isnull()[Null_Cols[i]]].shape[0]
        Per_Nulls = Num_Nulls/dt_Frame.shape[0]
        if(Per_Nulls<=0.8):
            get_Cols.append(Null_Cols[i])
        else:
            cols_Remove.append(Null_Cols[i])
            
    return((get_Cols,cols_Remove))
    
    
 
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
    
os.chdir('C:/Users/tatsa02/Documents/Workspace-Python/ipython-notebooks-master/ipython-notebooks-master/data')
path = os.getcwd() + '/Training_Dataset_full.csv'
data = pd.read_csv(path,header='infer')

dropped=data.drop(['TARGET',], axis=1)

dropped=dropped.loc[dropped.FISCAL_PERIOD.str.contains('2017'),:]


get_Correlated_Pairs(X_train,0.3)

"*****************************Applying Model on 20% milestone data **************************


os.chdir('C:/Users/tatsa02/Documents/Workspace-Python/ipython-notebooks-master/ipython-notebooks-master/data')
path = os.getcwd() + '/Training_Dataset_full.csv'
data = pd.read_csv(path,header='infer')

DUMMYS=pd.get_dummies(data,columns=['GEO','ACCOUNT_SEGMENT','INDUSTRY','GBU1','BUSINESS_TYPE','INSIDE_OUTSIDE',
                                    'OPPORTUNITY_SOURCE'],prefix_sep='_')
DUMMYS=DUMMYS[DUMMYS.CURRENT_MILESTONE==2]

Training_data = DUMMYS.loc[DUMMYS.FISCAL_PERIOD.str.contains('2017'),:]

Training_data.groupby(['TARGET'])['CURRENT_MILESTONE'].count()

Training_data = Training_data.iloc[:,3:55]

#Corr_Pairs = get_Correlated_Pairs(Training_data.drop(['TARGET'], axis=1),0.5)

data_predictors_train=Training_data.drop(['TARGET'], axis=1)

data_target_train=Training_data['TARGET']


"*****************************Applying Model on test **************************
y_pred = classifier.predict(data_predictors_test)

y_pred = classifier.predict_proba(data_predictors_test)

y_pred1=[1 if y >0.7 else 0 for y in y_pred[:,1]]

import matplotlib.pyplot  as plt



from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix
import pylab as pl
confusion_matrix = confusion_matrix(data_target_test, y_pred)


from sklearn.metrics import recall_score,accuracy_score,precision_score

precision_score(data_target_test,y_pred)

print(confusion_matrix)

cm = confusion_matrix(y_test, pred)
pl.matshow(confusion_matrix)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()
print('Accuracy of logistic regression classifier on test set: {:.2f}'.
      format(classifier.score(data_predictors_test, data_target_test)))

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred))

"*****************************Applying Model on train **************************

y_pred_train = classifier.predict(data_predictors_train)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(data_target_train, y_pred_train)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(data_target_train, y_pred_train))
from sklearn.linear_model import LogisticRegression



"**********************AUC*******************

import sklearn.metrics as metrics

fpr_RF,tpr_RF,thresh_RF = metrics.roc_curve(data_target_train,classifier.predict_proba
                                            (data_predictors_train)[:,1])
auc = metrics.auc(fpr_RF,tpr_RF)

import matplotlib.pyplot as plt

plt.figure()
lw = 2
plt.plot(fpr_RF, tpr_RF, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show() 
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


os.chdir('C:/Users/tatsa02/Documents/Workspace-Python/ipython-notebooks-master/ipython-notebooks-master/data')
path = os.getcwd() + '/Training_Dataset_full.csv'
data = pd.read_csv(path,header='infer')

DUMMYS=pd.get_dummies(data,columns=['GEO','ACCOUNT_SEGMENT','INDUSTRY','GBU1','BUSINESS_TYPE','INSIDE_OUTSIDE',
                                    'OPPORTUNITY_SOURCE'],prefix_sep='_')
DUMMYS=DUMMYS[DUMMYS.CURRENT_MILESTONE==2]

Training_data = DUMMYS.loc[DUMMYS.FISCAL_PERIOD.str.contains('2017'),:]

Training_data.groupby(['TARGET'])['CURRENT_MILESTONE'].count()

Training_data = Training_data.iloc[:,3:55]

Training_data=Training_data.drop(['CA_ACCOUNT_ID','FISCAL_PERIOD'], axis=1)

#Corr_Pairs = get_Correlated_Pairs(Training_data.drop(['TARGET'], axis=1),0.5)

data_predictors_train=Training_data.drop(['TARGET'], axis=1)

data_target_train=Training_data['TARGET']

target = 'TARGET'
IDcol = 'OPPORTUNITY_NUMBER'

"*****************function*********************
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

"**********************************

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
predictors = [x for x in Training_data.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, Training_data, predictors)
os.chdir('C:/Users/tatsa02/Documents/Workspace-Python/ipython-notebooks-master/ipython-notebooks-master/data')
path = os.getcwd() + '/Testing_Dataset_2017-04-01.csv'
data = pd.read_csv(path,header='infer')
DUMMYS1=pd.get_dummies(data,columns=['GEO','ACCOUNT_SEGMENT','INDUSTRY','GBU1','BUSINESS_TYPE','INSIDE_OUTSIDE',
                                    'OPPORTUNITY_SOURCE'],prefix_sep='_')
Testing_data = DUMMYS1.loc[DUMMYS1.FISCAL_PERIOD.isin(pd.Series(['2018 1'])),:]
Testing_data=Testing_data.drop(['CA_ACCOUNT_ID','FISCAL_PERIOD','INSIDE_OUTSIDE_LIKELY'], axis=1)
data_predictors_test=Testing_data.drop(['TARGET','OPPORTUNITY_NUMBER'], axis=1)
data_target_test=Testing_data['TARGET']
y_pred=gbm0.predict(data_predictors_test)
from sklearn.metrics import accuracy_score,recall_score

accuracy_score(data_target_test,gbm0.predict(data_predictors_test))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(data_target_test, y_pred)
from sklearn.metrics import recall_score,accuracy_score,precision_score

precision_score(data_target_test,y_pred)
print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.
      format(gbm0.score(data_predictors_test, data_target_test)))

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred))
gbm0.predict_proba(data_predictors_test)
proba=gbm0.predict_proba(data_predictors_test)
[1 if y >0.7 else 0 for y in proba[:,1]]
y_pred1=[1 if y >0.7 else 0 for y in proba[:,1]]
from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))
y_pred1=[1 if y >0.6 else 0 for y in proba[:,1]]
from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))
y_pred1=[1 if y >0.4 else 0 for y in proba[:,1]]
confusion_matrix = confusion_matrix(data_target_test, y_pred1)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))
y_pred1=[1 if y >0.4 else 0 for y in proba[:,1]]
confusion_matrix = confusion_matrix(data_target_test, y_pred1)
type(y_pred)
type(y_pred1)
y_pred1=np.asarray(y_pred1)
confusion_matrix = confusion_matrix(data_target_test, y_pred1)
confusion_matrix = confusion_matrix(data_target_test, y_pred)
gbm0.predict(data_predictors_test)
from sklearn.metrics import accuracy_score,recall_score

accuracy_score(data_target_test,gbm0.predict(data_predictors_test))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(data_target_test, y_pred)
confusion_matrix = confusion_matrix(data_target_test, y_pred1)
type(y_pred1)
confusion_matrix = confusion_matrix(data_target_test, y_pred)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(data_target_test, y_pred)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(data_target_test, y_pred1)
print(confusion_matrix)
y_pred1=[1 if y >0.7 else 0 for y in proba[:,1]]

y_pred1=np.asarray(y_pred1)

from sklearn.metrics import confusion_matri

confusion_matrix = confusion_matrix(data_target_test, y_pred1)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))
y_pred1=[1 if y >0.7 else 0 for y in proba[:,1]]

y_pred1=np.asarray(y_pred1)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(data_target_test, y_pred1)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))
y_pred1=[1 if y >0.7 else 0 for y in proba[:,1]]

y_pred1=np.asarray(y_pred1)

from sklearn.metrics import confusion_matrix


confusion_matrix = confusion_matrix(data_target_test, y_pred1)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))
y_pred1=[1 if y >0.2 else 0 for y in proba[:,1]]

y_pred1=np.asarray(y_pred1)

from sklearn.metrics import confusion_matrix


confusion_matrix = confusion_matrix(data_target_test, y_pred1)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))
y_pred1=[1 if y >0.15 else 0 for y in proba[:,1]]

y_pred1=np.asarray(y_pred1)

from sklearn.metrics import confusion_matrix


confusion_matrix = confusion_matrix(data_target_test, y_pred1)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))
y_pred1=[1 if y >0.1 else 0 for y in proba[:,1]]

y_pred1=np.asarray(y_pred1)

from sklearn.metrics import confusion_matrix


confusion_matrix = confusion_matrix(data_target_test, y_pred1)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))
y_pred1=[1 if y >0.7 else 0 for y in proba[:,1]]

y_pred1=np.asarray(y_pred1)

from sklearn.metrics import confusion_matrix


confusion_matrix = confusion_matrix(data_target_test, y_pred1)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))
y_pred1=[1 if y >0.5 else 0 for y in proba[:,1]]

y_pred1=np.asarray(y_pred1)

from sklearn.metrics import confusion_matrix


confusion_matrix = confusion_matrix(data_target_test, y_pred1)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred1))
predictors = [x for x in Training_data.columns if x not in [target, IDcol]]
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',
                                subsample=0.8,random_state=10),param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
param_test1 = {'n_estimators':np.arange(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',
                                subsample=0.8,random_state=10),param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(Training_data[predictors],Training_data[target])
Training_data
Training_data[predictors]
Training_data[target]
gsearch1.fit(Training_data[predictors],Training_data[target])
if __name__ == '__main__'

gsearch1.fit(Training_data[predictors],Training_data[target])
predictors = [x for x in Training_data.columns if x not in [target, IDcol]]
gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=9, min_samples_split=1200,min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
modelfit(gbm_tuned_1, train, predictors)
predictors = [x for x in Training_data.columns if x not in [target, IDcol]]
gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=9, min_samples_split=1200,min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
modelfit(gbm_tuned_1, Training_data, predictors)
y_pred=gbm_tuned_1.predict(data_predictors_test)
from sklearn.metrics import accuracy_score,recall_score

accuracy_score(data_target_test,gbm_tuned_1.predict(data_predictors_test))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(data_target_test, y_pred1)
confusion_matrix
predictors = [x for x in Training_data.columns if x not in [target, IDcol]]
gbm_tuned_2 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600,max_depth=9, min_samples_split=1200,min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
modelfit(gbm_tuned_1, Training_data, predictors)
y_pred=gbm_tuned_2.predict(data_predictors_test)

from sklearn.metrics import accuracy_score,recall_score

accuracy_score(data_target_test,gbm_tuned_2.predict(data_predictors_test))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(data_target_test, y_pred1)
from sklearn.metrics import recall_score,accuracy_score,precision_score

precision_score(data_target_test,y_pred)
y_pred=gbm_tuned_2.predict(data_predictors_test)
modelfit(gbm_tuned_1, Training_data, predictors)
modelfit(gbm_tuned_2, Training_data, predictors)
y_pred=gbm_tuned_2.predict(data_predictors_test)
from sklearn.metrics import accuracy_score,recall_score

accuracy_score(data_target_test,gbm_tuned_2.predict(data_predictors_test))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(data_target_test, y_pred)
from sklearn.metrics import recall_score,accuracy_score,precision_score
precision_score(data_target_test,y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred))
predictors = [x for x in Training_data.columns if x not in [target, IDcol]]
gbm_tuned_3 = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1200,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7,
warm_start=True)
modelfit(gbm_tuned_3, Training_data, predictors, performCV=False)
y_pred=gbm_tuned_3.predict(data_predictors_test)
from sklearn.metrics import accuracy_score,recall_score

accuracy_score(data_target_test,gbm_tuned_3.predict(data_predictors_test))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(data_target_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import recall_score,accuracy_score,precision_score

precision_score(data_target_test,y_pred)
from sklearn.metrics import classification_report
print(classification_report(data_target_test, y_pred))
predictors = [x for x in Training_data.columns if x not in [target, IDcol]]
gbm_tuned_4 = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1500,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7,
warm_start=True)
modelfit(gbm_tuned_4, Training_data, predictors, performCV=False)
y_pred=gbm_tuned_4.predict(data_predictors_test)
from sklearn.metrics import accuracy_score,recall_score
accuracy_score(data_target_test,gbm_tuned_4.predict(data_predictors_test))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(data_target_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import recall_score,accuracy_score,precision_score

precision_score(data_target_test,y_pred)
from sklearn.metrics import recall_score,accuracy_score,precision_score,recall_score
recall_score(data_target_test,y_pred)




for col in ['AGE_AT_1','AGE_AT_3','AGE_AT_4','AGE_AT_5','CUM_AGE_AT_1','CUM_AGE_AT_3','CUM_AGE_AT_4','CUM_AGE_AT_5']:
    percentiles = data_predictors_train[col].quantile([0.01,0.99]).values
    data_predictors_train[col][data_predictors_train[col] <= percentiles[0]] = percentiles[0]
    data_predictors_train[col][data_predictors_train[col] >= percentiles[1]] = percentiles[1]

























 