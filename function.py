from pyhive import hive
import pandas as pd
import numpy as np
import sys
import math
import random
import os
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ks_2samp

#0 read data
def read_data(sql,conn,path):
    
    start_time = time.time()   
    if os.path.exists(path):
        df=pd.read_csv(path,encoding='utf-8')
    else:
        df=pd.read_sql(sql, conn)
        df.rename(columns=lambda x: x.split('.')[-1], inplace=True)
        df.loc[:,'RandNum'] = [random.random() for i in range(len(df))]
        df.to_csv(path, index=False, sep=',', header=True,encoding='utf-8')      
    end_time = time.time()
    print ("读数据时间:%.2f minutes" % ((end_time-start_time)*1.0/60))
    print ("数据长度: %d" % len(df))
    return df


def ks_table(real,pred,bins=20):
    bad=pd.DataFrame(real,columns=['bad'])
    score=pd.DataFrame(pred,columns=['score'])
    data=pd.concat([bad,score],axis=1)
    data['good'] = 1 - data.bad
    data['bucket'] = pd.qcut(data.score,bins,duplicates='drop')
    grouped = data.groupby('bucket', as_index = False)
    table = pd.DataFrame(grouped.min().score, columns = ['min_score'])
    table['max_score'] = grouped.max().score
    table['min_score'] = grouped.min().score
    table['bads'] = grouped.sum().bad
    table['goods'] = grouped.sum().good
    table['total'] = table.bads + table.goods
    table['tot_ratio'] = (table.total / table.total.sum()).apply('{0:.2%}'.format)
    table = (table.sort_values(by = 'min_score',ascending=False)).reset_index(drop = True)
    table['odds'] = (table.bads / table.goods).apply('{0:.2f}'.format)
    table['bad_rate'] = (table.bads / table.total).apply('{0:.2%}'.format)
    table['lift'] = ((table.bads / table.total) / (table.bads.sum()/table.total.sum())).apply('{0:.2f}'.format)
    table['cum_bad'] = ((table.bads / data.bad.sum()).cumsum()).apply('{0:.2%}'.format)
    table['ks'] = np.round(((table.bads / data.bad.sum()).cumsum() - (table.goods / data.good.sum()).cumsum()), 4) * 100
    flag = lambda x: '<<<<<<' if x == table.ks.max() else ''
    table['max_ks'] = table.ks.apply(flag)
    print (table[['total','min_score','max_score','bads','bad_rate','cum_bad','lift','ks','max_ks']])
    return table


def ks_auc(real,pred,bins=20):
    bad=pd.DataFrame(real,columns=['bad'])
    score=pd.DataFrame(pred,columns=['score'])
    data=pd.concat([bad,score],axis=1)
    data['good'] = 1 - data.bad
    data['bucket'] = pd.qcut(data.score,bins,duplicates='drop')
    grouped = data.groupby('bucket', as_index = False)
    table = pd.DataFrame(grouped.min().score, columns = ['min_score'])
    table['max_score'] = grouped.max().score
    table['min_score'] = grouped.min().score
    table['bads'] = grouped.sum().bad
    table['goods'] = grouped.sum().good
    table = (table.sort_values(by = 'min_score',ascending=False)).reset_index(drop = True)
    table['ks'] = np.round(((table.bads / data.bad.sum()).cumsum() - (table.goods / data.good.sum()).cumsum()), 4) * 100
    ks_max = table.ks.max()
    auc = metrics.roc_auc_score(real,pred)
    return (ks_max,auc)


def PerformanceCheck(alg,InputData,Predicators,Target,bins=20):
    
    Temp=InputData.loc[(InputData[Target].isin([0,1])),:]
    Prob=pd.DataFrame(alg.predict_proba(Temp[Predicators])[:, 1],columns=['ProbScore'],index=Temp[Predicators].index)
    TempScore=pd.concat([Temp[['order_original_id','Group','s1d30','s3d30','apply_time','apply_month']],Prob],axis=1)
    
    group_ks_df = pd.DataFrame(columns = ['Group','KS_'+Target,'AUC_'+Target])
    index=0
    for group in sorted(list(set(TempScore['Group']))):
        if TempScore.loc[TempScore['Group']==group,:][Target].sum()>0:
            data_set = TempScore.loc[TempScore['Group']==group,:]
            (ks,auc) = ks_auc(list(data_set[[Target]].values),list(data_set[['ProbScore']].values),bins)
            group_ks_df.loc[index] = [group,ks,auc]
            index += 1
    print (group_ks_df)
    
    month_ks_df = pd.DataFrame(columns = ['Month','KS_'+Target,'AUC_'+Target])
    index = 0
    for month in sorted(list(set(TempScore['apply_month']))):
        if TempScore.loc[TempScore['apply_month']==month,:][Target].sum()>0:
            data_set = TempScore.loc[TempScore['apply_month']==month,:]
            (ks,auc) = ks_auc(list(data_set[[Target]].values),list(data_set[['ProbScore']].values),bins)
            month_ks_df.loc[index] = [month,ks,auc]
            index += 1
    month_ks_df.loc[:,'Month']=month_ks_df['Month'].astype('int64')
    print (month_ks_df)


def PerformanceCheck2(alg,InputData,Predicators,Target,bins=20):
    
    Temp=InputData.loc[(InputData[Target[0]].isin([0,1])),:]
    Prob=pd.DataFrame(alg.predict_proba(Temp[Predicators])[:, 1],columns=['ProbScore'],index=Temp[Predicators].index)
    TempScore=pd.concat([Temp[['order_original_id','Group','s1d30','s3d30','apply_time','apply_month']],Prob],axis=1)
    
    group_ks_df=pd.DataFrame()
    for target in Target:
        ks_list = []
        auc_list = []
        group_list =[]
        for group in sorted(list(set(TempScore['Group']))):
            if TempScore.loc[TempScore['Group']==group,:][target].sum()>0:
                data_set = TempScore.loc[TempScore['Group']==group,:]
                (ks,auc) = ks_auc(list(data_set[[target]].values),list(data_set[['ProbScore']].values),bins)
                ks_list.append(ks)
                auc_list.append(auc)
                group_list.append(group)
        ks_df=pd.DataFrame(ks_list,columns=['KS_'+target],index=group_list)
        auc_df=pd.DataFrame(auc_list,columns=['AUC_'+target],index=group_list)
        group_ks_df=pd.concat([group_ks_df,ks_df,],axis=1)
        group_ks_df=pd.concat([group_ks_df,auc_df,],axis=1)        
    print (group_ks_df)
    
    month_ks_df = pd.DataFrame()
    for target in Target:
        ks_list = []
        auc_list = []
        group_list =[]
        for group in sorted(list(set(TempScore['apply_month']))):
            if TempScore.loc[TempScore['apply_month']==group,:][target].sum()>0:
                data_set = TempScore.loc[TempScore['apply_month']==group,:]
                (ks,auc) = ks_auc(list(data_set[[target]].values),list(data_set[['ProbScore']].values),bins)
                ks_list.append(ks)
                auc_list.append(auc)
                group_list.append(group)
        ks_df=pd.DataFrame(ks_list,columns=['KS_'+target],index=group_list)
        auc_df=pd.DataFrame(auc_list,columns=['AUC_'+target],index=group_list)
        month_ks_df=pd.concat([month_ks_df,ks_df,],axis=1)
        month_ks_df=pd.concat([month_ks_df,auc_df,],axis=1)        
    print (month_ks_df)
    

def PerformanceCheck3(alg,InputData,Predicators,Target,Group,bins=20):
    
    Temp=InputData.loc[(InputData[Target[0]].isin([0,1])),:]
    Prob=pd.DataFrame(alg.predict_proba(Temp[Predicators])[:, 1],columns=['ProbScore'],index=Temp[Predicators].index)
    TempScore=pd.concat([Temp[['order_original_id','Group','s1d30','s3d30','apply_time','apply_month']],Prob],axis=1)
    
    group_ks_df=pd.DataFrame()
    for target in Target:
        ks_list = []
        auc_list = []
        group_list =[]
        for i in Group:
            for group in sorted(list(set(TempScore[i]))):
                if TempScore.loc[TempScore[i]==group,:][target].sum()>0:
                    data_set = TempScore.loc[TempScore[i]==group,:]
                    (ks,auc) = ks_auc(list(data_set[[target]].values),list(data_set[['ProbScore']].values),bins)
                    ks_list.append(ks)
                    auc_list.append(auc)
                    group_list.append(group)
        ks_df=pd.DataFrame(ks_list,columns=['KS_'+target],index=group_list)
        auc_df=pd.DataFrame(auc_list,columns=['AUC_'+target],index=group_list)
        group_ks_df=pd.concat([group_ks_df,ks_df,],axis=1)
        group_ks_df=pd.concat([group_ks_df,auc_df,],axis=1)        
    return group_ks_df
    
    
#函数4：按数据集输出ks_table
def ks_table_store(alg,InputData,Predicators,Target,file_name,bins=10):
    
    Temp=InputData.loc[(InputData[Target].isin([0,1])),:]
    Prob=pd.DataFrame(alg.predict_proba(Temp[Predicators])[:, 1],columns=['ProbScore'],index=Temp[Predicators].index)
    TempScore=pd.concat([Temp[['order_original_id','Group','s1d30','s3d30','apply_time']],Prob],axis=1)
    ks_table_result = pd.DataFrame()
    for group in sorted(list(set(TempScore['Group']))):
        data_set = TempScore.loc[TempScore['Group']==group,:]
        if data_set[Target].sum()>0:
            ks_table_name = ("%s_table" % group)
            print ('KS Table for %s' % group)
            ks_table_name = ks_table(data_set[Target].values,data_set['ProbScore'].values,bins)
            ks_table_name.loc[:,'group'] = group
            ks_table_result=pd.concat([ks_table_result,ks_table_name],axis=0)
    
    ks_table_result.to_csv(file_name,index=False,sep=',',header=True)

    
def calculate_psi(RawAll,ColsAll,path,threshold):
    
    expected=RawAll[RawAll['Group']=='INS'][ColsAll].values
    RawEDD=pd.DataFrame()
    group_set = set(RawAll['Group'])
    group_set.remove('INS')
    psi_remove_feature = set()
    psi_remove_df=pd.DataFrame(columns=['set','psi'])
    
    for i in group_set:
        #基础统计
        tmp=pd.DataFrame(RawAll[RawAll['Group']==i].describe().T,)
    
        #PSI统计
        actual=RawAll[RawAll['Group']==i][ColsAll].values
        if len(expected.shape) == 1:
            psi_values = np.empty(len(expected.shape))
        else:
            psi_values = np.empty(expected.shape[1])
        for j in range(0, len(psi_values)):
            if len(psi_values) == 1:
                psi_values = psi(expected, actual, 10)
                if psi_values>= threshold:
                    psi_remove_feature.add(ColsAll)
                    psi_remove_df.loc[ColsAll]=[('INS -> %s' % i ),psi_values]
            else:
                psi_values[j] = psi(expected[:,j], actual[:,j], 10)
                if psi_values[j]>= threshold:
                    psi_remove_feature.add(ColsAll[j])
                    psi_remove_df.loc[ColsAll[j]]=[('INS -> %s' % i ),psi_values[j]]
                    
        psi_output=pd.DataFrame(psi_values, columns=['PSI'],index=ColsAll) 
        tmp=pd.concat([tmp,psi_output],axis=1)
    
        tmp.loc[:,'Group']=i
        RawEDD=pd.concat([RawEDD,tmp],axis=0)
        
    print (psi_remove_df.sort_values(by='psi',ascending=False))
    RawEDD.to_csv(path)
    return psi_remove_feature
    

# 计算各特征的psi
def feature_psi(train,test,model_features,path,threshold):
    
    index = 0
    psi_remove_feature = set()
    df = pd.DataFrame(columns = ['feature_name','psi_score'])
    psi_remove_df=pd.DataFrame(columns=['set','psi'])
    
    for feature in model_features:
        psi_score=psi(train[feature],test[feature])
        df.loc[index]=[feature,psi_score]
        index+=1
        if psi_score >= threshold:
            psi_remove_feature.add(feature)
            psi_remove_df.loc[feature]=['train -> test',psi_score]
            
    print (psi_remove_df.sort_values(by='psi',ascending=False))
    RawAll = df.sort_values(by='psi_score',ascending=False)
    RawAll.to_csv(path)
    return psi_remove_feature   
    

def psi_features_table(InputDf,BenchmarkVar,Benchmark,GroupVar,GroupVal,features):

    psi_df=pd.DataFrame()
    for i in GroupVal:
        psi_df_set = pd.DataFrame(columns=[i])
        for feature in features:
            expected = (InputDf.loc[InputDf[BenchmarkVar].isin(Benchmark),feature]).dropna(axis = 0)
            expected.dropna(axis = 0)
            actual = (InputDf.loc[InputDf[GroupVar]==i,feature]).dropna(axis = 0)
            psi_score = psi(expected,actual)
            psi_df_set.loc[feature]=round(psi_score,4)
        
        psi_df=pd.concat([psi_df,psi_df_set],axis=1)
    return psi_df    
    
    
# psi函数
def psi(expected_array, actual_array, buckets=10, buckettype='bins'):
    '''Calculate the PSI for a single variable 
    
    Args:
       expected_array: numpy array of original values
       actual_array: numpy array of new values, same size as expected
       buckets: number of percentile ranges to bucket the values into
       
    Returns:
       psi_value: calculated PSI value
        '''
        
    def scale_range (input, min, max):
        input += -(np.min(input))
        if max - min != 0:
            input /= np.max(input) / (max - min)
        input += min
        return input
    
    
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    
    if buckettype == 'bins':
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
    elif buckettype == 'quantiles':
        breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])
    
    def generate_counts(arr, breakpoints):
        '''Generates counts for each bucket by using the bucket values 
        
        Args:
           arr: ndarray of actual values
           breakpoints: list of bucket values
        
        Returns:
           counts: counts for elements in each bucket, length of breakpoints array minus one
        '''
    
        def count_in_range(arr, low, high, start):
            '''Counts elements in array between low and high values.
               Includes value if start is true
            '''
            if start:
                return(len(np.where(np.logical_and(arr>=low, arr<=high))[0]))
            return(len(np.where(np.logical_and(arr>low, arr<=high))[0]))
    
        
        counts = np.zeros(len(breakpoints)-1)
    
        for i in range(1, len(breakpoints)):
            counts[i-1] = count_in_range(arr, breakpoints[i-1], breakpoints[i], i==1)
    
        return(counts)
    
    
    expected_percents = generate_counts(expected_array, breakpoints) / len(expected_array)
    actual_percents = generate_counts(actual_array, breakpoints) / len(actual_array)
    
    def sub_psi(e_perc, a_perc):
        '''Calculate the actual PSI value from comparing the values.
           Update the actual value to a very small number if equal to zero
        '''
        if a_perc == 0:
            a_perc = 0.001
        if e_perc == 0:
            e_perc = 0.001
        
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return(value)
    
    psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

    return(psi_value)


def sample_distribution(df,label):
    
    distri_df = pd.DataFrame()
    GROUP = []
    TOTAl = []
    S1D30 = []
    S3D30 = []
    S1D30_RATIO = []
    S3D30_RATIO = []
    S3D30_S1D30_RATIO = []
    group = sorted(list(set(df[label])))
    for i in group:
        df_current = df.loc[df[label]==i,:]
        total_current = len(df_current)
        s1d30_current = df_current['s1d30'].sum()
        s3d30_current = df_current['s3d30'].sum()
        s1d30_ratio_current = str(np.round(s1d30_current/total_current,4)*100)+'%'
        s3d30_ratio_current = str(np.round(s3d30_current/total_current,4)*100)+'%'
        s3d30_s1d30 = np.round(s3d30_current/s1d30_current,2)
        
        GROUP.append(i)
        TOTAl.append(total_current)
        S1D30.append(s1d30_current)
        S3D30.append(s3d30_current)
        S1D30_RATIO.append(s1d30_ratio_current)
        S3D30_RATIO.append(s3d30_ratio_current)
        S3D30_S1D30_RATIO.append(s3d30_s1d30)
    
    distri_df.loc[:,'group'] = GROUP
    distri_df.loc[:,'total'] = TOTAl
    distri_df.loc[:,'s1d30'] = S1D30
    distri_df.loc[:,'s3d30'] = S3D30
    distri_df.loc[:,'s1d30_ratio'] = S1D30_RATIO
    distri_df.loc[:,'s3d30_ratio'] = S3D30_RATIO
    distri_df.loc[:,'s3d30/s1d30'] = S3D30_S1D30_RATIO
    
    return distri_df


def sample_distribution_total(df,label):
    
    distri_df = pd.DataFrame()
    GROUP = []
    TOTAl = []
    IS_LOANED = []
    LOAN_RATIO = []
    S1D30 = []
    S3D30 = []
    S1D30_RATIO = []
    S3D30_RATIO = []
    S3D30_S1D30_RATIO = []
    group = sorted(list(set(df[label])))
    for i in group:
        df_current = df.loc[df[label]==i,:]
        total_current = len(df_current)
        loaned_current = np.round(df_current['is_loaned'].sum(),0)
        loan_ratio_current = str(np.round(loaned_current/total_current*100,2))+'%'
        s1d30_current = df_current['s1d30'].sum()
        s3d30_current = df_current['s3d30'].sum()
        s1d30_ratio_current = str(np.round(s1d30_current/loaned_current*100,2))+'%'
        s3d30_ratio_current = str(np.round(s3d30_current/loaned_current*100,2))+'%'
        s3d30_s1d30 = np.round(s3d30_current/s1d30_current,2)
        
        GROUP.append(i)
        TOTAl.append(total_current)
        IS_LOANED.append(loaned_current)
        LOAN_RATIO.append(loan_ratio_current)                          
        S1D30.append(int(s1d30_current))
        S3D30.append(int(s3d30_current))
        S1D30_RATIO.append(s1d30_ratio_current)
        S3D30_RATIO.append(s3d30_ratio_current)
        S3D30_S1D30_RATIO.append(s3d30_s1d30)
    
    distri_df.loc[:,'group'] = GROUP
    distri_df.loc[:,'total'] = TOTAl
    distri_df.loc[:,'is_loaned'] = IS_LOANED
    distri_df.loc[:,'loan_ratio'] = LOAN_RATIO                              
    distri_df.loc[:,'s1d30'] = S1D30
    distri_df.loc[:,'s3d30'] = S3D30
    distri_df.loc[:,'s1d30_ratio'] = S1D30_RATIO
    distri_df.loc[:,'s3d30_ratio'] = S3D30_RATIO
    distri_df.loc[:,'s3d30/s1d30'] = S3D30_S1D30_RATIO
    
    return distri_df


def PSI_Calculate(InputDf,GroupVar,BenchmarkGroup,CompareGroup,Cols):
    
    expected=InputDf.loc[InputDf[GroupVar].isin(BenchmarkGroup),:][Cols].values
    Output=pd.DataFrame()
    for i in sorted(list(set(CompareGroup))):
        actual=InputDf[InputDf[GroupVar]==i][Cols].values
        if len(expected.shape) == 1:
            psi_values = np.empty(len(expected.shape))
        else:
            psi_values = np.empty(expected.shape[1])
        for j in range(0, len(psi_values)):
            if len(psi_values) == 1:
                psi_values = psi(expected, actual, 10)
            else:
                psi_values[j] = psi(expected[:,j], actual[:,j], 10)
        ColsName=str(i)+'_PSI'
        psi_output=pd.DataFrame(psi_values,columns=[ColsName],index=Cols)
        Output=pd.concat([Output,psi_output],axis=1)
    return Output

def KS_Calculate(InputDf,GroupVar,GroupValue,Cols,TargetVar):
    
    Output=pd.DataFrame()
    for Cols_i in set(Cols):
        OutputPre=pd.DataFrame()
        for Group_i in sorted(list(set(GroupValue))):
            ks_value=ks_2samp(InputDf.loc[(InputDf[TargetVar]==1)&(InputDf[GroupVar]==Group_i)][Cols_i], InputDf.loc[(InputDf[TargetVar]!=1)&(InputDf[GroupVar]==Group_i)][Cols_i]).statistic
            ColsName=Group_i+'_KS'
            ks_output=pd.DataFrame(ks_value,columns=[ColsName],index=[Cols_i])
            OutputPre=pd.concat([OutputPre,ks_output],axis=1)
        Output=pd.concat([Output,OutputPre],axis=0)
    return Output

def EDD_Calculate(InputDf,GroupVar,GroupValue,Cols):
    
    Output=pd.DataFrame()
    for Group_i in sorted(list(set(GroupValue))):
        OutputPre=pd.DataFrame()
        OutputPre=pd.DataFrame(InputDf[InputDf[GroupVar]==Group_i][Cols].describe().T,)
        OutputPre.loc[:,Group_i+'_mean']= OutputPre.loc[:,'mean']
        #OutputPre.loc[:,'TotalCount']=InputDf[InputDf[GroupVar]==Group_i][Cols].fillna(0).count()
        OutputPre.loc[:,GroupVar]=Group_i
        Output=pd.concat([Output,OutputPre[[Group_i+'_mean']]],axis=1)
    return Output