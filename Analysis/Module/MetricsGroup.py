import os
import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


from sklearn.metrics import silhouette_score
from lifelines import CoxPHFitter
import statsmodels.api as sm
from itertools import chain




#---------------------------------------------------------------------------------------------------------------------------###
def cohen_d(d1, d2):
    n1, n2 = len(d1), len(d2)
    v1, v2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    p_s = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    m1, m2 = np.mean(d1), np.mean(d2)
    return (m1 - m2) / p_s




def DoMetric (DataMetric, LayerList):
    InpInd, InpFeat, IndEmbeddWeig, FeatEmbeddWeig, IndCentroid, FeatCentroid, ICosCLSim, FCosCLSim = LayerList 
    Data, TTE, EVENT, NCL_Ind, NCL_Feat, NumGene_CL,IntToGene= DataMetric
    
   
    # Access to similarity values between patient vectors and cluster centroids and gene vectors and cluster centroids
    PredICosCLSim = Model(InpInd,ICosCLSim)(np.array([[1]])).numpy()
    PredFCosCLSim = Model(InpFeat,FCosCLSim)(np.array([[1]]))[1:].numpy()

    # Assigning patient vectors to corresponding clusters
    IndCentMembers = np.argmax(PredICosCLSim, axis=-1)
    IndCentCounts = np.array([np.sum(IndCentMembers==CLID) for CLID in range(NCL_Ind)])


    # Assigning gene vectors to corresponding clusters
    FeatGroupLables = np.argmax(PredFCosCLSim, axis=-1)
    UniqFeatGLabel = np.unique(FeatGroupLables, return_counts=True)[0]
    FeatCentCounts = np.unique(FeatGroupLables, return_counts=True)[1]


    for i in range(NCL_Feat):
        if i not in UniqFeatGLabel:
            FeatCentCounts = np.insert(FeatCentCounts, i, 0).copy() 

    # Selecting gene names for the post-hoc analysis
    SelectedGeneID = []
    for TCL in range(NCL_Feat):
        FeatGroupLoc = np.where(FeatGroupLables==TCL)[0]
        FeatGroupLocInd = np.argsort(PredFCosCLSim[FeatGroupLables==TCL, TCL])[-NumGene_CL:]
        SelectedGeneID.append(FeatGroupLoc[FeatGroupLocInd] + 1)

    # Convert integer to gene name
    SelectedGeneName = []
    for i in SelectedGeneID:
        SelectedGeneName.append([IntToGene[j] for j in i])
    SelectedGeneNameList = np.unique(list(chain.from_iterable(SelectedGeneName)))

    # Clinical data processing for post-hoc analysis   
    ClinicData = Data[['patient_id','tumor_type','time','event']]
    SelGeneData = Data[SelectedGeneNameList]
    SelectedDATA = pd.concat([ClinicData, SelGeneData], axis=1)
    SelectedDATA['event'] = SelectedDATA['event'].astype('int32')
    PostHocSet =  SelectedDATA.iloc[:, 1:].copy()
    PostHocSet['IndCentMembers'] = IndCentMembers # The patient embedding matrix has already been sorted.
    UniqTumorType = np.sort(PostHocSet['tumor_type'].unique())
    IndSizeCancer = []
    for idx in UniqTumorType:
        UniqInd = np.unique(IndCentMembers[PostHocSet['tumor_type'] == idx], return_counts=True)
        UniqIndMask = np.array([ i in UniqInd[0] for i in np.arange(NCL_Ind) ])
        IndSizeCancer.append(UniqIndMask[None] * UniqInd[1])
    IndSizeCancer = np.concatenate(IndSizeCancer, axis=0)   
    IndCentRatio = min(np.min(IndSizeCancer, axis=1) / np.max(IndSizeCancer, axis=1))

    # Performing t-test between groups and obtaining p-values, adjust p-values, and cohen's d values
    # Performing survival anaylsis obtaining p-values and coefficient values
    SelectedGeneNameList = np.sort(SelectedGeneNameList)

    pRes = []
    AdjpRes = []
    tRes =[]
    CoDRes = []
    ABSSurvCoefRes = []
    SurvpValRes = []


    for typeTumor in UniqTumorType:
        TypeSet = PostHocSet[PostHocSet['tumor_type'] == typeTumor].copy()
        CoxHMSet = TypeSet[['time','event','IndCentMembers']]

        SubpRes = []
        SubtRes = []
        SubCoDRes = []

        # Performing survival analysis and obtaining p-value
        try:
            cph = CoxPHFitter()
            cph.fit(CoxHMSet, duration_col='time', event_col='event')
            ABSSurvCoefRes.append( np.abs(cph.summary['coef'][0]))  
            SurvpValRes.append( np.round(cph.summary['p'][0], 3)) 
            
            if cph.summary.coef[0] < 0: 
                TypeSet.loc[TypeSet['tumor_type'] == typeTumor, 'IndCentMembers'] -= 1
                TypeSet.loc[TypeSet['tumor_type'] == typeTumor, 'IndCentMembers'] = TypeSet.loc[TypeSet['tumor_type'] == typeTumor, 'IndCentMembers']**2
            
        except:
            ABSSurvCoefRes.append( 1e-7)  
            SurvpValRes.append(1.)
            
        for num, gene in enumerate (SelectedGeneNameList) :
            
            if num % 100 ==0:
                print(num)
            
            Group0 = TypeSet[TypeSet['IndCentMembers'] == 0][gene]
            Group1 = TypeSet[TypeSet['IndCentMembers'] == 1][gene]
            SubpRes.append(sm.stats.ttest_ind(Group0, Group1,usevar='unequal' )[1]) # p-value
            SubCoDRes.append(cohen_d(Group0,Group1)) # cohen's d

        pRes.append(SubpRes)
        AdjpRes.append(sm.stats.multipletests(SubpRes,alpha=0.1, method='fdr_bh')[1])
        tRes.append(SubtRes)
        CoDRes.append(SubCoDRes)


    PRate = [ np.sum(np.array(subLis) < 0.05, axis=-1) / len(subLis) for subLis in pRes]
    AdjPRate = [ np.sum(np.array(subLis) < 0.05, axis=-1) / len(subLis) for subLis in AdjpRes]
    MeanABSCohD = [ np.mean(np.abs(subLis)) for subLis in CoDRes]
    MeanCohD = [ np.mean(subLis) for subLis in CoDRes]
    
    
    AdjpRes = np.array(AdjpRes)
    CoDRes = np.array(CoDRes)
    
    SigMask = AdjpRes < 0.05
    FiltCoDRes= CoDRes * SigMask
    NegMask = FiltCoDRes < 0
    PosMask = FiltCoDRes > 0
    NegSigRate = np.sum(NegMask, axis=-1)/NegMask.shape[-1]
    PosSigRate = np.sum(PosMask, axis=-1)/PosMask.shape[-1]
    
    

    
    # Metrics for selecting the best model
    AvgtPRate = np.mean(PRate)
    AvgtAdjPRate = np.mean(AdjPRate) 
    MintAdjPRate = np.min(AdjPRate) 
    
    AvgABSGeCohD = np.mean(MeanABSCohD)
    MinABSGeCohD = np.min(MeanABSCohD)

    AvgABSSurvCoef = np.mean(ABSSurvCoefRes)    
    MinABSSurvCoef = np.min(ABSSurvCoefRes)    
    
    AvgSurvpVal = np.mean(SurvpValRes)    
    MaxSurvpVal = np.max(SurvpValRes)    
    
    AvgNegSigRate = np.mean(NegSigRate)
    MinNegSigRate = np.min(NegSigRate )
    
    AvgPosSigRate = np.mean(PosSigRate)
    MinPosSigRate = np.min(PosSigRate )
    


    return [AvgtPRate, AvgtAdjPRate, MintAdjPRate, AvgABSGeCohD, MinABSGeCohD, AvgABSSurvCoef, MinABSSurvCoef, AvgSurvpVal, MaxSurvpVal,
            np.exp(-AvgSurvpVal),np.exp(-MaxSurvpVal), AvgNegSigRate,  MinNegSigRate, AvgPosSigRate, MinPosSigRate, IndCentRatio], [NegSigRate, PosSigRate,SurvpValRes]
    # [AdjPRate, ABSSurvCoefRes] for visualization of umap



def DoAggMetric (AggMetricList, MetricTable):
    

    MetricTable = MetricTable.reset_index(drop=True)    

    # Normalization of metrics
    MinMetric =  MetricTable[AggMetricList].min()
    MaxMetric =  MetricTable[AggMetricList].max()
    NormMetric = ((MetricTable[AggMetricList] - MinMetric)/(MaxMetric-MinMetric)) 
    NormMetric.columns = ['Norm'+i for i in AggMetricList]
    MetricTable = pd.concat([MetricTable, NormMetric.fillna(0)],axis=1)

    # Calculating the aggregated metric to select the best model. 
    # AvgtPRate is not used to calculate the metric since it is proportional to tAdjP
    MetricTable['Metrics'] = np.sum(MetricTable[[i for i in MetricTable.columns if 'Norm' in i]], axis=1)
    
    return MetricTable


def DoSimEval (MetricTable, pFilter, pCutoff, MetricList, ExcRate, NmodEachG):
    
    ListGroupM = np.unique(MetricTable['GroupM'])
    AggMetricTable = pd.DataFrame()
    
    for GM in ListGroupM:
        print(GM)

        # 0. Model group selection
        SelGroup = MetricTable[MetricTable['GroupM'] == GM].copy()
        
        # 1. pFilter based filter out
        SelMetric = SelGroup[SelGroup[pFilter] < pCutoff].copy()
        
        if len(SelMetric) < NmodEachG:
            SelMetric = SelGroup.copy()
        
        # 2. MetricList based filter out
        for metric in MetricList:
            SelMetric = SelMetric.sort_values(metric)[int(SelMetric.shape[0] * ExcRate):].copy()
            print('N obs with filter of ' +str(metric)+ ' :' , SelMetric.shape[0])

        
        ## Aggregated Metrics
        SelMetric = DoAggMetric(MetricList, SelMetric).sort_values('Metrics')[-NmodEachG:]
        
        AggMetricTable = AggMetricTable.append(SelMetric)
        print()
        
    return AggMetricTable.reset_index(drop=True)
