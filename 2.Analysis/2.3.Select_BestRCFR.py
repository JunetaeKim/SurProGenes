# Parameters for post-hoc models; you must set those parameters for this task
ModelID = 'M03' # Model ID
NumGene_CL = 100 # The max number of genes to select for evaluation, denoted as Kn in the manuscript.
pCutoff = 0.005 # COX hazard model significance criteria to select learning results during priority-based model selection.
ExcRate = 0.2 # Percentage of results to be excluded during priority-based model selection.
NmodEahG = 1 # The number of best models to select for each independent learning during priority-based model selection.


# Path setting
FilePath = './ModelResults/'
SavePath = './EvalResults/'
ModelName = 'RCFR'


import pickle
import os
import sys
import pandas as pd
import numpy as np
import re

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model ,load_model

from lifelines import CoxPHFitter
from Models.RCFR import SetModel
from Module.DataProcessing import DataLoad
from Module.MetricsGroup import DoMetric, DoAggMetric, DoSimEval


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



## function for priority-based model selection
def Aggregation(MetricTable,AggMetricList):
    AggMetricTable = DoSimEval(MetricTable, 'MaxSurvpVal',pCutoff, AggMetricList, ExcRate, NmodEahG)
    AggMetricRank = DoAggMetric(AggMetricList, AggMetricTable[['Model']+AggMetricList]).sort_values('Metrics')
    AggMetricRank = pd.merge(AggMetricRank, AggMetricTable[['Model','MaxSurvpVal']], on='Model', how='left')
    BestModel = AggMetricRank.sort_values('Metrics').iloc[-1]
    
    return AggMetricRank, BestModel


# Model Preset; the parameter values must be the same as in the model training step.
EmbedSize = 50
NCL_Feat = 5
NCL_Ind = 2



if __name__ == "__main__":
    
    ## Data load
    StackedData, IntToGene, TTE, EVENT, TrIndEmbeddMask, ReferencePatIDLong, ReferencePatIDShort, NormDismInd, MergedData= DataLoad()

    PatIDX = StackedData[:, 0:1].astype('int')
    GeneIDX = StackedData[:, 1:2].astype('int')
    GeneExp = StackedData[:, 2:3]

    IndN = len(np.unique(PatIDX))
    FeatN = len(np.unique(GeneIDX))

    
    # Task set-up
    ModelList = os.listdir(FilePath)
    ModelList = [i for i in ModelList if ModelID in i ]


    # Model structure load
    RCFR, LayerList = SetModel( NormDismInd, TrIndEmbeddMask, IndN, FeatN, ReferencePatIDLong, ReferencePatIDShort)

    # Data for calculating metric
    DataMetric = [MergedData, TTE, EVENT, NCL_Ind, NCL_Feat, NumGene_CL, IntToGene]

    ColList = ['Model','AvgtPRate', 'AvgtAdjPRate', 'MintAdjPRate', 'AvgABSGeCohD', 'MinABSGeCohD', 'AvgABSSurvCoef', 'MinABSSurvCoef', 'AvgSurvpVal', 
               'MaxSurvpVal', 'NegExpAvgSurvpVal', 'NegExpMinSurvpVal', 'AvgNegSigRate',  'MinNegSigRate', 'AvgPosSigRate', 'MinPosSigRate','IndCentRatio']


    
    ## Procedure for model evaluation
    MetricTable = pd.DataFrame(columns=ColList)
    InfoFeatGroupList = []

    for num, model in enumerate(ModelList[:]):
        print(num)

        RCFR.load_weights(FilePath + model)  # Model weights load
        InpInd, InpFeat, IndEmbeddWeig, IndEmbeddReferenceLong, FeatEmbeddWeig, IndCentroid, FeatCentroid, ICosCLSim, FCosCLSim = LayerList

        # Metric calculation: InfoFeatGroup will be used in UMAP analysis
        metrics, InfoFeatGroup = DoMetric (DataMetric, [InpInd, InpFeat, IndEmbeddWeig, FeatEmbeddWeig, IndCentroid, FeatCentroid, ICosCLSim, FCosCLSim])
        InfoFeatGroupList.append(InfoFeatGroup)
        print('NegSigRate :',InfoFeatGroup[0],' , PosSigRate :',InfoFeatGroup[1],' , SurvpVal :',InfoFeatGroup[2])
        MetricTable = pd.concat([MetricTable, pd.DataFrame([[model] + metrics], columns=ColList)], axis=0)

    MetricTable['GroupM'] = np.array([re.findall('.\d+', i)[1][1:] for i in  MetricTable['Model']])
    MetricTable['EpNum'] = np.array([ re.findall('.\d+\.', i)[0][1:-1] for i in  MetricTable['Model']]).astype('int')
    MetricTable = MetricTable.sort_values(['GroupM','EpNum'])

    # Saving the metric table
    MetricTable.to_csv(SavePath+ModelName+'_MetricTable_Filt'+str(NumGene_CL)+'.csv',index=False)

       
    ## Procedure for priority-based model selection by metrics
    NegMetricList = ['IndCentRatio', 'MinABSSurvCoef', 'AvgABSSurvCoef',  'MinNegSigRate', 'AvgNegSigRate', 'MinABSGeCohD', 'AvgABSGeCohD']
    PosMetricList = ['IndCentRatio', 'MinABSSurvCoef', 'AvgABSSurvCoef', 'MinPosSigRate', 'AvgPosSigRate', 'MinABSGeCohD', 'AvgABSGeCohD']

    NegAggMetricRank, NegBestModel =  Aggregation(MetricTable, NegMetricList)
    PosAggMetricRank, PosBestModel =  Aggregation(MetricTable, PosMetricList)

    NegAggMetricRank.to_csv(SavePath+ModelName+'_Neg_AggMetricRank_Filt'+str(NumGene_CL)+'.csv',index=False)
    PosAggMetricRank.to_csv(SavePath+ModelName+'_Pos_AggMetricRank_Filt'+str(NumGene_CL)+'.csv',index=False)



    

    
