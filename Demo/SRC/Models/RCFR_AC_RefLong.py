import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model ,load_model
from tensorflow.keras.layers import Input, Dense,Concatenate, Reshape, Activation, BatchNormalization,Flatten, Embedding, Dot, Dropout


def SetModel (AdjCosWeight_, ExogDTL, ExogPRM, IndN, FeatN, ReferencePatIDLong, ReferencePatIDShort):
   
    
    ### I wrote the code in a hacky way to make debugging easier. That is, this hacky code allows you to treat an abstract weight matrix as a class variable like a keras symbolic tensor.    
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
            input = K.sum(input) * 0 + 1 # To return only the weight matrix, make the previous input tensor value a scalar 1 and multiply it with the weight matrix. It's just a hacky trick
            return (input*self.GenVec)

    
    ### I wrote the code in a hacky way to make debugging easier. That is, this hacky code allows you to treat an abstract weight matrix as a class variable like a keras symbolic tensor.    
    class SuppleVec(tf.keras.layers.Layer):

        def __init__(self, Info):
            super(SuppleVec, self).__init__()

        def build(self, input_shape):
            self.GenVec1 = tf.constant(ExogDTL, name='gen1')
            self.GenVec2 = tf.constant(ExogPRM, name='gen2')

        def call(self, input):
            input = K.sum(input) * 0 + 1 # To return only the weight matrix, make the previous input tensor value a scalar 1 and multiply it with the weight matrix. It's just a hacky trick
            return (input*self.GenVec1, input*self.GenVec2)    

    
    
    
    
    
    
    #------------------------------------------------ Model Parameters -------------------------------------------------------------------------
    
    EmbedSize = 50
    NCL_Feat = 5
    NCL_Ind = 2
    
    XI1 = 0.2 #ξ_1
    XI2 = 1. #ξ_2
    XI3 = 0.2 #ξ_3
    XI4 = 1. #ξ_4
    XI5 = 0.002 #ξ_5
    
    Top_k=2
    
    AdjTheta = 1. + AdjCosWeight_ *0.1 # theta
    
    
    
    #------------------------------------------------ Model part 1 -------------------------------------------------------------------------

    InpInd = Input(shape=(1,))
    DTL, PRM = SuppleVec([])(InpInd)

    PEM = tf.transpose(DoGenVec([EmbedSize, IndN + 1 ])(InpInd))
    PEMNorm = tf.linalg.l2_normalize(PEM[1:], axis=-1)

    PCM = DoGenVec([NCL_Ind, EmbedSize ])(InpInd)
    PCMNorm = tf.linalg.l2_normalize(PCM, axis=-1)



    # CHLoss_pat 
    ICosCLSim = tf.matmul( PEMNorm, PCMNorm, transpose_b=True)
    ICosTheta = tf.acos(K.clip(ICosCLSim, -1.+K.epsilon(), 1.0-K.epsilon()))
    ICosCLDist = ICosTheta/np.pi
    IMinCLDist = tf.reduce_min(ICosCLDist, axis=-1, keepdims=True)
    CHL_pat = tf.reduce_mean(tf.maximum(IMinCLDist-XI2, K.epsilon()))

    # RLoss_pat
    IREmbeddAssoSim =  tf.matmul(PEMNorm, PEMNorm, transpose_b=True)
    IREmbeddTheta = tf.acos(K.clip(IREmbeddAssoSim, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
    PED = IREmbeddTheta/np.pi
    RL_pat = tf.reduce_mean(PRM*(PED - DTL )**2)  


    
    #------------------------------------------------ Model part 2 -------------------------------------------------------------------------

    PEMBatch = tf.nn.embedding_lookup(PEM[:], tf.cast(InpInd, dtype=tf.int32))[:,0]
    PEMBatchNorm = tf.linalg.l2_normalize(PEMBatch, axis=-1) 

    InpFeat = Input(shape=(1,))
    GEM = tf.transpose(DoGenVec([EmbedSize, FeatN + 1])(InpFeat))
    GEMNorm = tf.linalg.l2_normalize(GEM[:], axis=-1)
    GCM = DoGenVec([NCL_Feat, EmbedSize ])(InpFeat)

    GEMBatch = tf.nn.embedding_lookup(GEM[:], tf.cast(InpFeat, dtype=tf.int32))[:,0]
    GEMBatchNorm = tf.linalg.l2_normalize(GEMBatch, axis=-1) 



    # Referencing for ACAM
    IndEmbeddReferenceLong = tf.nn.embedding_lookup(PEM[1:], tf.cast([[ReferencePatIDLong]], dtype=tf.int32))[:,0]
    IndEmbeddReferenceLongNorm = tf.linalg.l2_normalize(IndEmbeddReferenceLong, axis=-1)
    RefLongCosSim = tf.matmul((GEMNorm), tf.stop_gradient(IndEmbeddReferenceLongNorm), transpose_b=True)
    RefLongCosSim = K.clip(RefLongCosSim, -1.+K.epsilon(), 1.-K.epsilon())
    GLD = tf.acos(K.clip(RefLongCosSim, -1.+K.epsilon(), 1.- K.epsilon()))/np.pi
   
    IndEmbeddReferenceShort = tf.nn.embedding_lookup(PEM[1:], tf.cast([[ReferencePatIDShort]], dtype=tf.int32))[:,0]
    IndEmbeddReferenceShortNorm = tf.linalg.l2_normalize(IndEmbeddReferenceShort, axis=-1)
    RefShortCosSim = tf.matmul((GEMNorm), tf.stop_gradient(IndEmbeddReferenceShortNorm), transpose_b=True)
    RefShortCosSim = K.clip(RefShortCosSim, -1.+K.epsilon(), 1.-K.epsilon())
    RefShortCosTheta = tf.acos(K.clip(RefShortCosSim, -1.+K.epsilon(), 1.-K.epsilon()))/np.pi

    
    # GLDBar
    SoftDir = 0.5 - GLD
    DifSImLongShort = tf.abs(RefLongCosSim - RefShortCosSim)
    Magn = tf.exp(-(tf.exp(DifSImLongShort)))
    DeltaTheta = Magn * AdjTheta * SoftDir
    GLDBar = GLD + DeltaTheta

    # clustering loss COS
    GCMNorm = tf.linalg.l2_normalize(GCM, axis=-1)
    FCosCLSim = tf.matmul(GEMNorm, GCMNorm, transpose_b=True)
    FCosTheta = tf.acos(K.clip(FCosCLSim, -1.+K.epsilon(), 1.-K.epsilon()))
    FCosCLDist = FCosTheta/np.pi
    FMinCLDist = tf.reduce_min(FCosCLDist, axis=-1, keepdims=True)
    CHL_gene = tf.reduce_mean(tf.maximum(FMinCLDist-XI1, K.epsilon()))

     
    ## Gene out
    FCosCLSimBatch = tf.nn.embedding_lookup(FCosCLSim, tf.cast(InpFeat, dtype=tf.int32))[:,0]
    Z = Dot(1,1)([PEMBatchNorm , GEMBatchNorm])  * tf.reduce_max(FCosCLSimBatch, keepdims=True, axis=-1) + K.epsilon()
    WGX_hat = 0.5 + Z*0.5
    WGX_hat = Reshape((1,), name='OutVal')(WGX_hat) 


    # Angles between feature vectors
    FeatEmbeddAssoSim =  tf.matmul(GEMNorm, GEMNorm, transpose_b=True)
    FeatEmbeddAssoSim = K.clip(FeatEmbeddAssoSim, -1.+K.epsilon(), 1.-K.epsilon())
    FeatEmbeddAssoTheta = tf.acos(FeatEmbeddAssoSim)
 
    # Arccosine angle adjustment mechanism (ACAM)
    GLDXY = tf.matmul(GLDBar, GLDBar, transpose_b=True)
    GLD2 = tf.reduce_sum(GLDBar**2, axis=-1)
    GLDDist = tf.maximum(GLD2[:,None] + GLD2[None] -2*GLDXY, K.epsilon()) 
    GLDBar_eud = tf.math.sqrt(GLDDist)
    GED = FeatEmbeddAssoTheta/np.pi
    RL_gene = tf.reduce_mean( tf.maximum((GED - GLDBar_eud)**2 - 0.025, K.epsilon()) ) 
    

    ## Constrastive loss COS
    FDistBtwCL = tf.matmul(GCMNorm, GCMNorm, transpose_b=True)
    FDistBtwCL = tf.acos(K.clip(FDistBtwCL, -1.+K.epsilon(), 1.-K.epsilon()))/np.pi
    FSecTensors =  -tf.nn.top_k(-FDistBtwCL, k=Top_k, name='topk')[0][:, 1]
    SPL_gene = tf.reduce_mean(tf.maximum(XI3 - FSecTensors, K.epsilon()))

    IDistBtwCL = tf.matmul(PCMNorm, PCMNorm, transpose_b=True)
    IDistBtwCL = tf.acos(K.clip(IDistBtwCL, -1.+K.epsilon(), 1.-K.epsilon()))/np.pi
    ISecTensors =  -tf.nn.top_k(-IDistBtwCL, k=Top_k, name='topk')[0][:, 1]
    SPL_pat = tf.reduce_mean(tf.maximum(XI4 - ISecTensors, K.epsilon()))
    
    
    # Balnce regularizer
    MeanICosCLSim = tf.reduce_mean(ICosCLSim, axis=0,keepdims=True)
    DiffMeanICosCLSim = tf.maximum(( (MeanICosCLSim - tf.transpose(MeanICosCLSim))**2) - XI5, K.epsilon())
    BalanceICosCL =tf.reduce_sum(DiffMeanICosCLSim) / ((NCL_Ind)**2-NCL_Ind)
    Balance = BalanceICosCL 
    

    RunModel = Model([InpInd,InpFeat], WGX_hat)


    RunModel.add_loss(CHL_pat)
    RunModel.add_metric(CHL_pat, 'CHL_pat')

    RunModel.add_loss(CHL_gene)
    RunModel.add_metric(CHL_gene, 'CHL_gene')


    RunModel.add_loss(SPL_gene)
    RunModel.add_metric(SPL_gene, 'SPL_gene')

    RunModel.add_loss(SPL_pat)
    RunModel.add_metric(SPL_pat, 'SPL_pat')


    RunModel.add_loss(RL_pat)
    RunModel.add_metric(RL_pat, 'RL_pat')

    RunModel.add_loss(RL_gene)
    RunModel.add_metric(RL_gene, 'RL_gene')
    
    RunModel.add_loss(Balance)
    RunModel.add_metric(Balance, 'Balance')
    

    return RunModel, [InpInd, InpFeat, PEM, IndEmbeddReferenceLong, GEM, PCM, GCM, ICosCLSim, FCosCLSim]







    

    