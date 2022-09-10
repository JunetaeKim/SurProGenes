import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model ,load_model
from tensorflow.keras.layers import Input, Dense,Concatenate, Reshape, Activation, BatchNormalization,Flatten, Embedding, Dot, Dropout





def SetModel (AdjCosWeight_, NormDismInd, TrIndEmbeddMask, IndN, FeatN, ReferencePatIDLong, ReferencePatIDShort):
    
    
    
    class DoGenVec(tf.keras.layers.Layer):

        def __init__(self, Info):
            super(DoGenVec, self).__init__()
            self.NumCl = Info[0]
            self.EmbedDim = Info[1]

        def get_config(self):

            config = super().get_config().copy()
            config.update({'EmbedDim': self.EmbedDim , 'NumCl': self.NumCl })
            return config

        def build(self, input_shape):
            np.random.seed(1)
            self.GenVec = tf.Variable(np.random.normal(0, 0.1, size=(self.NumCl, self.EmbedDim)).astype(np.float32), trainable=True, name='GenVec')

        def call(self, input):
            input = K.sum(input) * 0 + 1 # Weight matrix만 return 해주기 위해 이전 input tensor 값을 스칼라 1로 변경 후 weight matrix와 곱함,
            return (input*self.GenVec)

    
    
    class SuppleVec(tf.keras.layers.Layer):

        def __init__(self, Info):
            super(SuppleVec, self).__init__()

        def build(self, input_shape):
            self.GenVec1 = tf.constant(NormDismInd, name='gen1')
            self.GenVec2 = tf.constant(TrIndEmbeddMask, name='gen2')

        def call(self, input):
            input = K.sum(input) * 0 + 1 # Weight matrix만 return 해주기 위해 이전 input tensor 값을 스칼라 1로 변경 후 weight matrix와 곱함,
            return (input*self.GenVec1, input*self.GenVec2)    

    
    
    
    
    
    
    #------------------------------------------------ Model Parameters -------------------------------------------------------------------------
    
    EmbedSize = 50
    NCL_Feat = 5
    NCL_Ind = 2
    FThresDist = 0.2
    FThresUpper = 0.2
    IThresDist = 1.
    IThresUpper = 1.
    Top_k=2
    
    AdjCosWeight = 1. + AdjCosWeight_ *0.1
    
    
    
    #------------------------------------------------ Model part 1 -------------------------------------------------------------------------

    InpInd = Input(shape=(1,))
    NormDismIndVec, TrIndEmbeddMaskVec = SuppleVec([])(InpInd)

    IndEmbeddWeig = tf.transpose(DoGenVec([EmbedSize, IndN + 1 ])(InpInd))
    IndEmbeddWeigNorm = tf.linalg.l2_normalize(IndEmbeddWeig[1:], axis=-1)

    IndCentroid = DoGenVec([NCL_Ind, EmbedSize ])(InpInd)
    IndCentroidNorm = tf.linalg.l2_normalize(IndCentroid, axis=-1)



    # Indivisual Cosine similarity 
    ICosCLSim = tf.matmul( IndEmbeddWeigNorm, IndCentroidNorm, transpose_b=True)
    ICosTheta = tf.acos(K.clip(ICosCLSim, -1.+K.epsilon(), 1.0-K.epsilon()))
    ICosCLDist = ICosTheta/np.pi
    IMinCLDist = tf.reduce_min(ICosCLDist, axis=-1, keepdims=True)
    CLCost_Ind = tf.reduce_mean(tf.maximum(IMinCLDist-IThresUpper, K.epsilon()))

    # Indivisual Risk by cosine angle
    IREmbeddAssoSim =  tf.matmul(IndEmbeddWeigNorm, IndEmbeddWeigNorm, transpose_b=True)
    IREmbeddTheta = tf.acos(K.clip(IREmbeddAssoSim, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
    IRSimLoss = tf.reduce_mean(TrIndEmbeddMaskVec*(IREmbeddTheta/np.pi - NormDismIndVec )**2)  


    
    #------------------------------------------------ Model part 2 -------------------------------------------------------------------------

    IndEmbeddWeigBatch = tf.nn.embedding_lookup(IndEmbeddWeig[:], tf.cast(InpInd, dtype=tf.int32))[:,0]
    IndEmbeddWeigBatchNorm = tf.linalg.l2_normalize(IndEmbeddWeigBatch, axis=-1) 

    InpFeat = Input(shape=(1,))
    FeatEmbeddWeig = tf.transpose(DoGenVec([EmbedSize, FeatN + 1])(InpFeat))
    FeatEmbeddWeigNorm = tf.linalg.l2_normalize(FeatEmbeddWeig[:], axis=-1)
    FeatCentroid = DoGenVec([NCL_Feat, EmbedSize ])(InpFeat)

    FeatEmbeddWeigBatch = tf.nn.embedding_lookup(FeatEmbeddWeig[:], tf.cast(InpFeat, dtype=tf.int32))[:,0]
    FeatEmbeddWeigBatchNorm = tf.linalg.l2_normalize(FeatEmbeddWeigBatch, axis=-1) 



    # Referencing
    IndEmbeddReferenceLong = tf.nn.embedding_lookup(IndEmbeddWeig[1:], tf.cast([[ReferencePatIDLong]], dtype=tf.int32))[:,0]
    IndEmbeddReferenceLongNorm = tf.linalg.l2_normalize(IndEmbeddReferenceLong, axis=-1)
    IndEmbeddReferenceShort = tf.nn.embedding_lookup(IndEmbeddWeig[1:], tf.cast([[ReferencePatIDShort]], dtype=tf.int32))[:,0]
    IndEmbeddReferenceShortNorm = tf.linalg.l2_normalize(IndEmbeddReferenceShort, axis=-1)


    RefLongCosSim = tf.matmul((FeatEmbeddWeigNorm), tf.stop_gradient(IndEmbeddReferenceLongNorm), transpose_b=True)
    RefLongCosSim = K.clip(RefLongCosSim, -1.+K.epsilon(), 1.-K.epsilon())
    RefLongCosTheta = tf.acos(K.clip(RefLongCosSim, -1.+K.epsilon(), 1.- K.epsilon()))/np.pi

    RefShortCosSim = tf.matmul((FeatEmbeddWeigNorm), tf.stop_gradient(IndEmbeddReferenceShortNorm), transpose_b=True)
    RefShortCosSim = K.clip(RefShortCosSim, -1.+K.epsilon(), 1.-K.epsilon())
    RefShortCosTheta = tf.acos(K.clip(RefShortCosSim, -1.+K.epsilon(), 1.-K.epsilon()))/np.pi


    Bound = 0.5 - RefLongCosTheta
    DifSImLongShort = tf.abs(RefLongCosSim - RefShortCosSim)
    ThetaBoundFunc = tf.exp(-(tf.exp(DifSImLongShort))) * AdjCosWeight
    DeltaTheta = ThetaBoundFunc * Bound
    AdjRefLongCosTheta = RefLongCosTheta + DeltaTheta

    # clustering loss COS
    FeatCentroidNorm = tf.linalg.l2_normalize(FeatCentroid, axis=-1)
    FCosCLSim = tf.matmul(FeatEmbeddWeigNorm, FeatCentroidNorm, transpose_b=True)
    FCosTheta = tf.acos(K.clip(FCosCLSim, -1.+K.epsilon(), 1.-K.epsilon()))
    FCosCLDist = FCosTheta/np.pi
    FMinCLDist = tf.reduce_min(FCosCLDist, axis=-1, keepdims=True)
    CLCost_Feat = tf.reduce_mean(tf.maximum(FMinCLDist-FThresUpper, K.epsilon()))

     
    ## Gene out
    FCosCLSimBatch = tf.nn.embedding_lookup(FCosCLSim, tf.cast(InpFeat, dtype=tf.int32))[:,0]

    Z = Dot(1,1)([IndEmbeddWeigBatchNorm , FeatEmbeddWeigBatchNorm])  * tf.reduce_max(FCosCLSimBatch, keepdims=True, axis=-1) + K.epsilon()

    ## Gene out
    OutVal = 0.5 + Z*0.5
    OutVal = Reshape((1,), name='OutVal')(OutVal) 



    # angles between feature vectors
    FeatEmbeddAssoSim =  tf.matmul(FeatEmbeddWeigNorm, FeatEmbeddWeigNorm, transpose_b=True)
    FeatEmbeddAssoSim = K.clip(FeatEmbeddAssoSim, -1.+K.epsilon(), 1.-K.epsilon())
    FeatEmbeddAssoTheta = tf.acos(FeatEmbeddAssoSim)

    RiskRanking = AdjRefLongCosTheta
    USXY = tf.matmul(RiskRanking, RiskRanking, transpose_b=True)
    US2 = tf.reduce_sum(RiskRanking**2, axis=-1)
    USDist = tf.maximum(US2[:,None] + US2[None] -2*USXY, K.epsilon()) 
    USSqrtDist = tf.math.sqrt(USDist)
    USSimLoss = tf.reduce_mean( tf.maximum((FeatEmbeddAssoTheta/np.pi - USSqrtDist)**2 - 0.025, K.epsilon()) ) 
    

    ## Constrastive loss COS
    FLossBtwCL = tf.matmul(FeatCentroidNorm, FeatCentroidNorm, transpose_b=True)
    FLossBtwCL = tf.acos(K.clip(FLossBtwCL, -1.+K.epsilon(), 1.-K.epsilon()))/np.pi
    FSecTensors =  -tf.nn.top_k(-FLossBtwCL, k=Top_k, name='topk')[0][:, 1]
    FWidLoss = tf.reduce_mean(tf.maximum(FThresDist - FSecTensors, K.epsilon()))

    ILossBtwCL = tf.matmul(IndCentroidNorm, IndCentroidNorm, transpose_b=True)
    ILossBtwCL = tf.acos(K.clip(ILossBtwCL, -1.+K.epsilon(), 1.-K.epsilon()))/np.pi
    ISecTensors =  -tf.nn.top_k(-ILossBtwCL, k=Top_k, name='topk')[0][:, 1]
    IWidLoss = tf.reduce_mean(tf.maximum(IThresDist - ISecTensors, K.epsilon()))
    
    
    # Balnce regularizer
    MeanICosCLSim = tf.reduce_mean(ICosCLSim, axis=0,keepdims=True)
    DiffMeanICosCLSim = tf.maximum(( (MeanICosCLSim - tf.transpose(MeanICosCLSim))**2) - 0.002, K.epsilon())
    BalanceICosCL =tf.reduce_sum(DiffMeanICosCLSim) / ((NCL_Ind)**2-NCL_Ind)
    Balance = BalanceICosCL 
    

    RunModel = Model([InpInd,InpFeat], OutVal)


    RunModel.add_loss(CLCost_Ind)
    RunModel.add_metric(CLCost_Ind, 'CLCost_Ind')

    RunModel.add_loss(CLCost_Feat)
    RunModel.add_metric(CLCost_Feat, 'CLCost_Feat')


    RunModel.add_loss(FWidLoss)
    RunModel.add_metric(FWidLoss, 'FWidLoss')

    RunModel.add_loss(IWidLoss)
    RunModel.add_metric(IWidLoss, 'IWidLoss')


    RunModel.add_loss(IRSimLoss)
    RunModel.add_metric(IRSimLoss, 'IRSimLoss')

    RunModel.add_loss(USSimLoss)
    RunModel.add_metric(USSimLoss, 'USSimLoss')
    
    RunModel.add_loss(Balance)
    RunModel.add_metric(Balance, 'Balance')
    

    return RunModel, [InpInd, InpFeat, IndEmbeddWeig, IndEmbeddReferenceLong, FeatEmbeddWeig, IndCentroid, FeatCentroid, ICosCLSim, FCosCLSim]







    

    