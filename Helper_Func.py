# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:28:29 2018

@author: janas02
"""

import pandas as pd
import numpy as np

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
    

def Get_Numeric_Columns(Dt_Frame):
    All_Cols =  Dt_Frame.columns.values.tolist()
    get_Non_Objects = []
    for i in range(len(All_Cols)):
        if(((Dt_Frame[[All_Cols[i]]].dtypes)[[All_Cols[i]]] == object)[All_Cols[i]]==False):
            if(np.var(Dt_Frame[[All_Cols[i]]])[All_Cols[i]]!=0):
                get_Non_Objects.append(All_Cols[i])
    return(get_Non_Objects)

def Get_Category_Columns(Dt_Frame):
    All_Cols =  Dt_Frame.columns.values.tolist()
    get_Non_Objects = []
    for i in range(len(All_Cols)):
        if(((Dt_Frame[[All_Cols[i]]].dtypes)[[All_Cols[i]]] == object)[All_Cols[i]]==True):
            get_Non_Objects.append(All_Cols[i])
    return(get_Non_Objects)

def Get_disttribution(dt_frame):
    columns_ = dt_frame.columns.values.tolist()
    dict_1 = {}
    percentile_ =[0,5,10,25,50,75,90,95,99,100]
    for i in columns_:
        col = dt_frame[~(dt_frame[i].isnull())][i]
        dict_1[i] = np.percentile(col,percentile_)
    Numeric_Dist = pd.DataFrame(dict_1,index=percentile_)
    return(Numeric_Dist)



def get_Cuts(dt_Frame,perc_Avg_Ovrl_Acct):
    cuts = []
    off_setFlag = False
    boundaries_List = []
    for i in range(len(perc_Avg_Ovrl_Acct)-1):
        if((perc_Avg_Ovrl_Acct[i]==perc_Avg_Ovrl_Acct[i+1]) & (perc_Avg_Ovrl_Acct[i]== dt_Frame.inputs.max())):
            cuts.append(dt_Frame.inputs.max())
            break            
        if(perc_Avg_Ovrl_Acct[i]==perc_Avg_Ovrl_Acct[i+1]):
            higher_ = dt_Frame[dt_Frame.inputs>perc_Avg_Ovrl_Acct[i]].sort_values(['inputs'],ascending=True).reset_index().inputs[0]
            cuts.append(perc_Avg_Ovrl_Acct[i])
            cuts.append(higher_)
            boundaries_List = [perc_Avg_Ovrl_Acct[i],higher_]
            off_set = higher_
            off_setFlag = True
            cuts.extend(boundaries_List)
        elif(off_setFlag):
            boundaries_List = [off_set,perc_Avg_Ovrl_Acct[i]]
            cuts.extend(boundaries_List)
        else:
            boundaries_List = [perc_Avg_Ovrl_Acct[i],perc_Avg_Ovrl_Acct[i+1]]
            cuts.extend(boundaries_List)
    cuts.append(dt_Frame.inputs.max())        
    Bin_ = list(set(cuts))
    Bin_ = sorted(Bin_,reverse=False)
    return(Bin_)


def Bin_Variable(input_,output_):
    dt_Frame = pd.concat([input_,output_],axis=1)
    Bin_Variable_ = input_.name
    dt_Frame.columns = ['inputs','outputs']
    dt_Frame = dt_Frame.loc[~(dt_Frame.inputs.isnull()),['inputs','outputs']]
    bins = []
    Total_Num = []
    Total_Percent = []
    Num_Of_Goods = []
    Num_Of_Bads = []
    percent_Good = []
    percent_bad = []
    WOE = []
    Difference = []
    IV = []
    Preserve_Numeric = 0
#    Offset_Flag = False
    Dir_chang = 0
    fix_first = 'neutral'
    perc_Input = np.percentile(dt_Frame.inputs,list(np.linspace(0,100,21,endpoint=True)))
    perc_Avg_Ovrl_Acct = get_Cuts(dt_Frame,perc_Input)
    for i in range(len(perc_Avg_Ovrl_Acct)-1):
        if(i==0):
            Series_ = dt_Frame.loc[(dt_Frame.inputs>=perc_Avg_Ovrl_Acct[i]) &(dt_Frame.inputs<=perc_Avg_Ovrl_Acct[i+1])]
            bin_ = str(perc_Avg_Ovrl_Acct[i])+'-'+str(perc_Avg_Ovrl_Acct[i+1])
        else:
            Series_ = dt_Frame.loc[(dt_Frame.inputs>perc_Avg_Ovrl_Acct[i]) &(dt_Frame.inputs<=perc_Avg_Ovrl_Acct[i+1])]
            bin_ = str(perc_Avg_Ovrl_Acct[i])+'-'+str(perc_Avg_Ovrl_Acct[i+1])
##        print(type(Series_))
        Churn_Series = Series_
##        print(Churn_Series)
##        Churn_Series.name = 'outputs'
        Tot_Rows = Churn_Series.shape[0]
        Num_OfGoods = Churn_Series[Churn_Series.outputs==1].shape[0]
        Num_OfBads = Churn_Series[Churn_Series.outputs==0].shape[0]
        Total_Num.append(Tot_Rows)
        TotPercent = Tot_Rows/dt_Frame.shape[0]
        Total_Percent.append(TotPercent)
        Num_Of_Goods.append(Num_OfGoods)
        Num_Of_Bads.append(Num_OfBads)
        Total_Goods = len(dt_Frame.outputs[dt_Frame.outputs==1])
        Total_Bads = len(dt_Frame.outputs[dt_Frame.outputs==0])
        if((Num_OfGoods==0)|(Num_OfBads==0)):
            Num_OfGoods = Num_OfGoods+0.5
            Num_OfBads = Num_OfBads+0.5
    #        print(Num_OfGoods)
    #        print(Num_OfBads)
        prob_Good = Num_OfGoods/Total_Goods
        prob_Bad = Num_OfBads/Total_Bads
        percent_Good.append(prob_Good)
        percent_bad.append(prob_Bad)
        Weight_Of_Evidence = np.log(prob_Good/prob_Bad)
        if(i==0):
            if(Weight_Of_Evidence>0):
                fix_first = 'pos'
            else:
                fix_first = 'neg'
        else:
            if(Weight_Of_Evidence>0):
                if(fix_first!='pos'):
                    Dir_chang = Dir_chang + 1
                    fix_first = 'pos'
            elif(Weight_Of_Evidence<0):
                if(fix_first!='neg'):
                    Dir_chang = Dir_chang + 1
                    fix_first = 'neg'

        WOE.append(Weight_Of_Evidence)
        Diff = prob_Good - prob_Bad
        Difference.append(Diff)
        IV.append(Diff * Weight_Of_Evidence)
        bins.append(bin_)
    if(Dir_chang==1):
        Preserve_Numeric=1
    else:
        Preserve_Numeric = 0        
    Dict_1 = {'bins':bins,'Total_Num':Total_Num,'Total_Percent':Total_Percent,'Num_Of_Goods':Num_Of_Goods,'Num_Of_Bads':Num_Of_Bads,'percent_Good':percent_Good,'percent_bad':percent_bad,
             'WOE':WOE,'Difference':Difference,'IV':IV,'Numeric_Flag':Preserve_Numeric}
    IV_Table = pd.DataFrame(Dict_1)
    IV_Table = IV_Table[['bins','Total_Num','Total_Percent','Num_Of_Goods','Num_Of_Bads','percent_Good',
                        'percent_bad','WOE','Difference','IV']]
    IV_Table['Variable'] = pd.Series(np.repeat(Bin_Variable_,IV_Table.shape[0]))
    IV_Table['Numeric_Flag'] = pd.Series(np.repeat(Preserve_Numeric,IV_Table.shape[0]))
    return(IV_Table)

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
	
def Evaluate_IV(inputs_,outputs_):
#    inputs_ = dt_Frame_Heart.Sex
#    outputs_ = dt_Frame_Heart.target
    dt_New = pd.concat([inputs_,outputs_],axis=1)
    Num_Of_Goods = []
    Num_Of_Bads = []
    Percent_Good = []
    Percent_Bad = []
    Total_Num =[]
    Percent_Total = []
    WOE =[]
    IV=[]
    Bin_Variable_ = inputs_.name
    dt_New.columns = ['inputs','outputs']
    for i in list(dt_New.inputs.unique()):
        filtered = dt_New.loc[dt_New.inputs==i,'outputs']
        Goods = len(filtered[filtered==1])
        Bads = len(filtered[filtered==0])
        Total = len(filtered)
        if((Goods==0)|(Bads==0)):
            Goods = Goods+0.5
            Bads = Bads+0.5
        Overall_Goods = len(dt_New.loc[dt_New.outputs==1,'outputs'])
        Overall_Bads = len(dt_New.loc[dt_New.outputs==0,'outputs'])
        Num_Of_Goods.append(Goods)
        Num_Of_Bads.append(Bads)
        Good_Per = Goods/Overall_Goods
        Bad_Per = Bads/Overall_Bads
        Percent_Good.append(Good_Per)
        Percent_Bad.append(Bad_Per)
        Total_Num.append(Total)
        Percent_Total.append(Total/dt_New.shape[0])
        WOE_ = np.log(Good_Per/Bad_Per)
        WOE.append(WOE_)
        Difference = (Goods/Overall_Goods)-(Bads/Overall_Bads)
        IV.append(Difference*WOE_)
    Dt_IV = pd.DataFrame({'Level':list(dt_New.inputs.unique()),'#ofGoods':Num_Of_Goods,'#ofBads':Num_Of_Bads,'Total_Per':Percent_Total,
                         'Perc_Good':Percent_Good,'Perc_Bad':Percent_Bad,'WOE':WOE,'IV':IV})
    Dt_IV['Variable'] = pd.Series(np.repeat(Bin_Variable_,Dt_IV.shape[0]))        
            
    
    return(Dt_IV)
        #    type(dt_New)
    ##    Num_Of_Goods = dt_New
	
