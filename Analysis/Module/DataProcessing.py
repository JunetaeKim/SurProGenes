import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf


def Restratification(RestratDATA):
    
    TUMOR_TYPE_COMBINATION = [ "BRCA", "GBM", "KICH", "KIRC", "KIRP", "LGG",]
    
    RestratDATA = RestratDATA[RestratDATA['tumor_type'].isin(TUMOR_TYPE_COMBINATION)].copy()
    
    
    RestratDATA.loc[ RestratDATA['tumor_type'] == "GBM", 'tumor_type' ] = "GLIOMA"
    RestratDATA.loc[ RestratDATA['tumor_type'] == "LGG", 'tumor_type' ] = "GLIOMA"
    RestratDATA.loc[ RestratDATA['tumor_type'] == "KIRP", 'tumor_type' ] = "KIPAN"
    RestratDATA.loc[ RestratDATA['tumor_type'] == "KICH", 'tumor_type' ] = "KIPAN"
    RestratDATA.loc[ RestratDATA['tumor_type'] == "KIRC", 'tumor_type' ] = "KIPAN"
    
   
    return RestratDATA


def DataLoad ():
    
    StackedData = np.load('../Data/ProcessedData/StakedgData_GroupNorm.npy', allow_pickle=True)
    IntToGene = np.load('../Data/ProcessedData/IntToGene_GroupNorm.npy', allow_pickle=True).tolist()
    DistMat = np.load('../Data/ProcessedData/DisimInd_GroupNorm.npy', allow_pickle=True)
    TTE = np.load('../Data/ProcessedData/TTE_GroupNorm.npy', allow_pickle=True)
    EVENT = np.load('../Data/ProcessedData/Event_GroupNorm.npy', allow_pickle=True)

    NegativeMask = ((TTE[:,None] - TTE[None])<0).astype('int')
    NegativeEvent = NegativeMask * EVENT[None]
    PositiveMask = ((TTE[:,None] - TTE[None])>0).astype('int')
    PositiveNonEvent = NegativeMask * (1-EVENT[None])
    TrIndEmbeddMask = (NegativeEvent + PositiveNonEvent).astype('float32')

    ReferencePatIDLong = np.argmax(TTE * (EVENT==0).astype('float32'))
    ReferencePatIDShort = np.where( (TTE == np.min(TTE [EVENT.astype('bool')]) ) & (EVENT==1))[0][0]

    NormDismInd = ((DistMat - DistMat.min()) / (DistMat.max() - DistMat.min())) 
    NormDismInd = tf.constant(NormDismInd, dtype=tf.float32)
    TrIndEmbeddMask = tf.constant(TrIndEmbeddMask, dtype=tf.float32)
    
    # load Merged data
    with open('../Data/ProcessedData/LogAnalData.pickle', 'rb') as f:
        LogAnalData = pickle.load(f)

    return StackedData, IntToGene, TTE, EVENT, TrIndEmbeddMask, ReferencePatIDLong, ReferencePatIDShort, NormDismInd, LogAnalData


