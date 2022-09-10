import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import random as python_random
from tensorflow.keras import backend as K


from tensorflow.keras.models import Model ,load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from Models.RCFR_AC_NOCL import SetModel

np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)

os.environ['TF_CUDNN_DETERMINISTIC'] = 'True'
os.environ['TF_DETERMINISTIC_OPS'] = 'True'

## GPU selection
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.99
# Create a session with the above options specified.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))     



    
    
if __name__ == "__main__":
    
    ## Data load

    # Data for the deep learning model
    StackedData = np.load('../Data/ProcessedData/StakedgData_GroupNorm.npy', allow_pickle=True)
    IntToGene = np.load('../Data/ProcessedData/IntToGene_GroupNorm.npy', allow_pickle=True).tolist()
    DistMat = np.load('../Data/ProcessedData/DisimInd_GroupNorm.npy', allow_pickle=True)
    TTE = np.load('../Data/ProcessedData/TTE_GroupNorm.npy', allow_pickle=True)
    EVENT = np.load('../Data/ProcessedData/Event_GroupNorm.npy', allow_pickle=True)

    # Masking vectors used for learning risk ordered embedding vectors
    NegativeMask = ((TTE[:,None] - TTE[None])<0).astype('int')
    NegativeNonEvent = NegativeMask * (1-EVENT[None])
    PositiveMask = ((TTE[:,None] - TTE[None])>0).astype('int')
    PositiveEvent = PositiveMask * EVENT[None]
    TrIndEmbeddMask = (NegativeNonEvent + PositiveEvent).astype('float32')

    # Setting the reference ID (Longest survivor vs shortest death)
    ReferencePatIDLong = np.argmax(TTE * (EVENT==0).astype('float32'))
    ReferencePatIDShort = np.where( (TTE == np.min(TTE [EVENT.astype('bool')]) ) & (EVENT==1))[0][0]

    # Data processing for deep learning model training
    NormDismInd = ((DistMat - DistMat.min()) / (DistMat.max() - DistMat.min())) 
    NormDismInd = tf.constant(NormDismInd, dtype=tf.float32)
    TrIndEmbeddMask = tf.constant(TrIndEmbeddMask, dtype=tf.float32)

    PatIDX = StackedData[:, 0:1].astype('int')
    GeneIDX = StackedData[:, 1:2].astype('int')
    GeneExp = StackedData[:, 2:3]

    IndN = len(np.unique(PatIDX))
    FeatN = len(np.unique(GeneIDX))

    
    AdjCosWeight_ = 0

    # Loop for doing simulation j times; "Deep learning results are often not reproducible"
    for j in range(1, 6):

        CosModel,_ = SetModel(AdjCosWeight_, NormDismInd, TrIndEmbeddMask, IndN, FeatN, ReferencePatIDLong, ReferencePatIDShort)
        Adam = tf.keras.optimizers.Adam( learning_rate=0.00025)
        CosModel.compile(loss='mse', optimizer=Adam, metrics={"OutVal":'mse' })

        MSavePoint = ModelCheckpoint(filepath='./Results/M04_RCFR_ACNOCL_W'+str(AdjCosWeight_)+'_S'+str(j)+ '_Epo'+'{epoch:02d}'+ ".hdf5", save_weights_only=True, monitor='loss', mode='min',  save_best_only=False)

        CosModel.fit(x=(PatIDX[:],GeneIDX[:]) , y=GeneExp[:], verbose=1, epochs=100, batch_size=250000, shuffle=True, callbacks = [MSavePoint] )

        
        
        
    
    
    

    

    
