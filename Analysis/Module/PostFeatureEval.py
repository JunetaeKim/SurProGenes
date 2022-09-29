from itertools import chain
import numpy as np
import pandas as pd
import os 
import warnings

from lifelines import CoxPHFitter
import statsmodels.api as sm

import venn
import umap

from tensorflow.keras.models import Model ,load_model
import tensorflow as tf
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns


#---------------------------------------------------------------------------------------------------------------------------###
def cohen_d(d1, d2):
    n1, n2 = len(d1), len(d2)
    v1, v2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    p_s = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    m1, m2 = np.mean(d1), np.mean(d2)
    return (m1 - m2) / p_s


class Components:
    
    def __init__ (self, SelModel, LayerList, Data, NCL_Ind, NCL_Feat, NumGene_CL,ReferencePatIDLong, ReferencePatIDShort, IntToGene,CancerSet=False):
        self.SelModel = SelModel
        self.LayerList = LayerList
        self.NumGene_CL = NumGene_CL
        self.ReferencePatIDLong = ReferencePatIDLong
        self.ReferencePatIDShort = ReferencePatIDShort
        self.CommonSetGene_run = False
        self.GeneNametoVecLoc = {v: k-1 for k, v in IntToGene.items()}
        self.NCL_Feat = NCL_Feat
        self.NCL_Ind = NCL_Ind
        
        
        if CancerSet :
            self.UniqueTumors= np.sort(CancerSet)
        else:
            self.UniqueTumors= np.sort(Data['tumor_type'].unique())
            
        if len(set(CancerSet)) != len(CancerSet):
            warnings.warn('Warning: Duplicate values exist in CancerSet')

        InpInd, InpFeat, IndEmbeddWeig, IndEmbeddReferenceLong, FeatEmbeddWeig, IndCentroid, FeatCentroid, ICosCLSim, FCosCLSim = LayerList
        

        # Assigning gene and individual vectors to corresponding clusters
        self.PredICosCLSim = Model(InpInd,ICosCLSim)([1]).numpy()
        self.PredFCosCLSim = Model(InpFeat,FCosCLSim)([1])[1:].numpy()
        
        self.PredIndCentroid = Model(InpInd,IndCentroid)([1]).numpy()
        self.PredFeatCentroid = Model([InpInd,InpFeat],FeatCentroid)(([1],[1])).numpy()
        
        self.IndCentMembers = np.argmax(self.PredICosCLSim, axis=-1)
        self.FeatGroupLables = np.argmax(self.PredFCosCLSim, axis=-1)
        
        self.PredIndEmbedd = Model(InpInd,IndEmbeddWeig)([1])[1:].numpy()
        self.PredFeatEmbedd = Model(InpFeat,FeatEmbeddWeig)([1])[1:].numpy()

        # Assigning gene and individual vectors to corresponding clusters
        self.IndCentMembers = np.argmax(self.PredICosCLSim, axis=-1)
        self.FeatGroupLables = np.argmax(self.PredFCosCLSim, axis=-1)

        # Calculating risk level
        PredIndEmbeddReferenceLong = Model(InpInd,IndEmbeddReferenceLong )([1]).numpy()
        PredIndEmbeddReferenceLongNorm = tf.linalg.l2_normalize(PredIndEmbeddReferenceLong, axis=-1)
        PredIndEmbeddNorm = tf.linalg.l2_normalize(self.PredIndEmbedd, axis=-1)
        IndRefCosSim = tf.matmul(PredIndEmbeddNorm, PredIndEmbeddReferenceLongNorm, transpose_b=True)
        IndRefCosSim = K.clip(IndRefCosSim, -1. + K.epsilon(), 1.-K.epsilon())
        self.IndRefTheta = tf.acos(IndRefCosSim).numpy()/np.pi
        
        
        # Selecting gene names for the post-hoc analysis
        SelectedGeneID = []
        for TCL in range(self.NCL_Feat):
            FeatGroupLoc = np.where(self.FeatGroupLables==TCL)[0]
            FeatGroupLocInd = np.argsort(self.PredFCosCLSim[self.FeatGroupLables==TCL, TCL])[-self.NumGene_CL:]
            #FeatGroupLocInd = np.argsort(self.PredFCosCLSim[self.FeatGroupLables==TCL, TCL])[:self.NumGene_CL] # debugg
            SelectedGeneID.append(FeatGroupLoc[FeatGroupLocInd] + 1)

        # Convert integer to gene name
        SelectedGeneName = []
        for i in SelectedGeneID:
            SelectedGeneName.append([IntToGene[j] for j in i])

        
        # Clinical data processing for post-hoc analysis   
        MergedData = Data[['patient_id','tumor_type','time','event']]
        SelGeneData = Data[np.unique(list(chain.from_iterable(SelectedGeneName)))]
        SelectedDATA = pd.concat([MergedData, SelGeneData], axis=1)
        SelectedDATA['event'] = SelectedDATA['event'].astype('int32')
        self.PostHocSet =  SelectedDATA.iloc[:, 1:].copy()
        self.PostHocSet['IndCentMembers'] = self.IndCentMembers # The patient embedding matrix has already been sorted.

        
        for idx, typeTumor in enumerate(self.UniqueTumors):
            CoxHMSet = self.PostHocSet[self.PostHocSet['tumor_type'] == typeTumor][['time','event','IndCentMembers']]

            # Conducting survival anaylsis for Kaplan-Meier curve
            cph = CoxPHFitter()
            cph.fit(CoxHMSet, duration_col='time', event_col='event')
            
            
            if cph.summary.coef[0] < 0: 
                self.PostHocSet.loc[self.PostHocSet['tumor_type'] == typeTumor, 'IndCentMembers'] -=1
                self.PostHocSet.loc[self.PostHocSet['tumor_type'] == typeTumor, 'IndCentMembers'] = self.PostHocSet.loc[self.PostHocSet['tumor_type'] == typeTumor, 'IndCentMembers']**2
            
        
        # Ind indivisual information
        IndInfo = self.PostHocSet[['tumor_type', 'IndCentMembers']].reset_index()
        self.IndCohortInfo = { i : IndInfo[IndInfo['tumor_type'] == i ] for i in self.UniqueTumors}  ## Class variable
        
        # Gene candidate list
        self.SelectedGeneNameList = np.sort(list(chain.from_iterable(SelectedGeneName)))  # Unlist gene list; Class variable
        
        
        
        
    def CommonSetGene (self, filt_metric='adjp', filt_cutpval=0.05, ):
        self.CommonSetGene_run = True
        
        ## Performing t-test between groups and obtaining p-values and adjust p-values 
        pRes = []
        AdjpRes = []
        tRes =[]
        CoDRes = []

        for typeTumor in self.UniqueTumors:
            TypeSet = self.PostHocSet[self.PostHocSet['tumor_type'] == typeTumor]
            
            SubpRes = []
            SubtRes = []
            SubCoDRes = []

            for gene in self.SelectedGeneNameList:
                Group0 = TypeSet[TypeSet['IndCentMembers'] == 0][gene]
                Group1 = TypeSet[TypeSet['IndCentMembers'] == 1][gene]
                
                SubtRes.append(sm.stats.ttest_ind(Group0, Group1,usevar='unequal' )[0]) # t-value
                SubpRes.append(sm.stats.ttest_ind(Group0, Group1,usevar='unequal' )[1]) # p-value
                SubCoDRes.append(cohen_d(Group0,Group1)) # cohen's d
               
            pRes.append(SubpRes)
            AdjpRes.append(sm.stats.multipletests(SubpRes,alpha=0.1, method='fdr_bh')[1])
            tRes.append(SubtRes)
            CoDRes.append(SubCoDRes)

        self.CoDRes = np.round(np.array(CoDRes), 3)
        self.AdjpRes = np.array(AdjpRes)
        
        SigMask = self.AdjpRes < filt_cutpval
        FiltCoDRes= CoDRes * SigMask

        self.NegMask = FiltCoDRes < 0
        self.PosMask = FiltCoDRes > 0
        
   
        # Filtering out insignificant variables     
        if filt_metric == 'adjp':
            ExGenes = [ np.array(self.SelectedGeneNameList)[subLis<filt_cutpval] for subLis in AdjpRes]
        elif filt_metric == 'p':
            ExGenes = [ np.array(self.SelectedGeneNameList)[subLis< filt_cutpval] for subLis in np.array(pRes)]
        self.ExGeneList = np.concatenate(ExGenes)
        self.ExGeneLocList = np.unique([self.GeneNametoVecLoc[i] for i in self.ExGeneList])


        ## Finding the intersection and drawing a Venn Diagram    
        self.CommonExGene= set.intersection(*[set(x) for x in ExGenes])
        self.ExGenesDict= {name : set(ExGenes[idx]) for idx, name  in enumerate (self.UniqueTumors )}

        self.ExGenesLocDict = {}
        for typeTumor in self.UniqueTumors:
            self.ExGenesLocDict[typeTumor] = [self.GeneNametoVecLoc[j] for j in self.ExGenesDict[typeTumor]]


        # Size information for each intersection
        VennStat = venn.generate_petal_labels([set(i) for i in ExGenes])


        ## Labeling for intersection Venn diagram and assiging information on the number of each set
        VennLabelsList = []
        for i in VennStat:
            VennLabelsList.append(np.array([j for j in i])[None])
        VennLabels = np.concatenate(VennLabelsList).astype('bool')

        self.VennLabelStat = {}
        self.UniqSetGeneList = {}
        for num, idx in enumerate (VennLabels):
            SetGroupLabel = ''
            for j in self.UniqueTumors[idx]:
                SetGroupLabel += '& '+j+' '
            self.VennLabelStat[SetGroupLabel[2:-1]] = int(list(VennStat.values())[num])

            Maskidx = (1 - idx).astype('bool')
            DiffCancer = [j for j in self.UniqueTumors[Maskidx]] 
            self.UniqSetGeneList[SetGroupLabel[2:-1]] = set.intersection(*[self.ExGenesDict[j] for j in self.UniqueTumors[idx]])

            if len(DiffCancer) >0:
                for i in DiffCancer:
                    self.UniqSetGeneList[SetGroupLabel[2:-1]] -= self.ExGenesDict[i]
        
        # Selecting unique expressed gene LOC by each set
        self.UniqSetGeneLoc = {}
        for i in self.UniqSetGeneList:
            self.UniqSetGeneLoc[i] = [self.GeneNametoVecLoc[j] for j in list(self.UniqSetGeneList[i])]

        VennLables = list(self.UniqSetGeneLoc.keys())
        self.VennLables = [ sorted(i.split(' & ')) for i in VennLables]    


        # Selecting commonly expressed unique gene for UMAP and visualization
        # Matching cancer set list to return key for UniqSetGeneLoc
        UniqSetKeyLoc = None
        for num, i in enumerate (self.VennLables):
            if len(self.UniqueTumors) == len(i) and self.UniqueTumors.tolist() == i:
                UniqSetKeyLoc = num     


        # Selecting unique expressed gene for UMAP and visualization
        self.CommonSetKey =list(self.UniqSetGeneLoc.keys())[UniqSetKeyLoc]        
        self.CommonGeneLoc = np.array(self.UniqSetGeneLoc[self.CommonSetKey])

        
        if (len(self.UniqueTumors)>=2) and (len(self.UniqueTumors)<=6): 
            venn.venn(self.ExGenesDict)
            print(self.VennLabelStat)
        else:
            warnings.warn('The number of categories for the intersection must be an integer between 2 and 6.')
            print(self.VennLabelStat)

            
    # SurvivalGraph------------------------------------------------------------------------------------------------------------------------------------------------------------------          
        
    def SurvivalGraph (self, mode='pooled',nrows=0,ncols=0):
        
        def PooledPlot (CoxHMSet):
            
            # Conducting survival anaylsis for Kaplan-Meier curve
            cph = CoxPHFitter()
            cph.fit(CoxHMSet, duration_col='time', event_col='event')

            '''
            if cph.summary.coef[0] > 0: 
                CLLabels = ['Low-risk group', 'High-risk group']
                palette = ['yellowgreen', 'red']

            else:
                CLLabels = ['High-risk group', 'Low-risk group']
                palette = ['red', 'yellowgreen']
            '''
            
            CLLabels = ['Low-risk group', 'High-risk group']
            palette = ['yellowgreen', 'red']
            
            # p and HR    
            pVal = round(cph.summary.p[0], 3)
            if pVal == 0:
                pVal = 'p < 0.001'
            else:
                pVal = 'p = ' + str(pVal)
            HR = 'H.R = '+str(round(np.exp(cph.summary.coef[0]), 3))

            plt.figure(figsize=(8,6))
            plt.plot(np.mean(cph.predict_survival_function(CoxHMSet[CoxHMSet['IndCentMembers']==0], times=range(5000)), axis=1), label=CLLabels[0], color=palette[0],linewidth=2.5)
            plt.plot(np.mean(cph.predict_survival_function(CoxHMSet[CoxHMSet['IndCentMembers']==1], times=range(5000)), axis=1), label=CLLabels[1], color=palette[1], linewidth=2.5)
            handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels

            CL1 = mlines.Line2D([], [], color='black', marker=None, linestyle='None',  linewidth=0., markersize=12,label=pVal)
            CL2 = mlines.Line2D([], [], color='black', marker=None, linestyle='None',  linewidth=0., markersize=12,label=HR)
            handles.extend([CL1]+[CL2])
            leg = plt.legend(handles=handles,prop={'size': 16},loc =1, framealpha=0.5)

            plt.xlabel('Days', fontsize = 17)
            plt.ylabel('% Surviving', fontsize = 18)
            plt.title('Pooled group across cancers', fontsize = 20)

            
        def EachPlot (PostHocSet,nrows,ncols):
          
            assert nrows * ncols >= len(self.UniqueTumors), '''ERROR: The size arguments for subplot, norws (number of rows) and ncols (number of columns) were not passed correctly. 
                Please provide the correct nrows and ncols parameters in arument.'''
            
            if nrows % 2 !=0:
                nrows += 1
            if ncols % 2 !=0:
                ncols += 1

            CLLabels = ['Low-risk group', 'High-risk group']
            palette = ['yellowgreen', 'red']
                
            plt.figure(figsize=(14,10))
            for idx, typeTumor in enumerate(self.UniqueTumors):
                CoxHMSet = PostHocSet[PostHocSet['tumor_type'] == typeTumor][['time','event','IndCentMembers']]

                # Conducting survival anaylsis for Kaplan-Meier curve
                cph = CoxPHFitter()
                cph.fit(CoxHMSet, duration_col='time', event_col='event')

                
                # p and HR    
                pVal = round(cph.summary.p[0], 3)
                if pVal == 0:
                    pVal = 'p < 0.001'
                else:
                    pVal = 'p = ' + str(pVal)
                HR = 'H.R = '+str(round(np.exp(cph.summary.coef[0]), 3))

                plt.subplot(nrows,ncols,idx+1)
                plt.subplots_adjust(hspace=0.35)
                plt.plot(np.mean(cph.predict_survival_function(CoxHMSet[CoxHMSet['IndCentMembers']==0], times=range(5000)), axis=1), label=CLLabels[0], color=palette[0],linewidth=2.5)
                plt.plot(np.mean(cph.predict_survival_function(CoxHMSet[CoxHMSet['IndCentMembers']==1], times=range(5000)), axis=1), label=CLLabels[1], color=palette[1], linewidth=2.5)
                handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels

                CL1 = mlines.Line2D([], [], color='black', marker=None, linestyle='None',  linewidth=0., markersize=12,label=pVal)
                CL2 = mlines.Line2D([], [], color='black', marker=None, linestyle='None',  linewidth=0., markersize=12,label=HR)
                handles.extend([CL1]+[CL2])
                #leg = plt.legend(handles=handles,prop={'size': 20},loc =1, framealpha=0.5)
                leg = plt.legend(handles=[CL1]+[CL2],prop={'size': 22},loc =1, framealpha=0.5)

                plt.xlabel('Days', fontsize = 22)
                plt.ylabel('% Surviving', fontsize = 22)
                plt.title(typeTumor, fontsize = 24)   



        if mode == 'pooled':
            PooledPlot(self.PostHocSet[['time','event','IndCentMembers']])
        elif mode == 'each':
            EachPlot(self.PostHocSet[['tumor_type','time','event','IndCentMembers']],nrows,ncols)
        else:
            Warning.warn("Wrong mode value, only mode values 'pooled', and 'each' are allowed.")

    
    # PerformUMPA-------------------------------------------------------------------------------------------------------------------------------------------------------------------  
        
    def PerformUMPA (self, SelGeneLoc=False, **UMAP_Parameters):
        
        self.UMPA_run = True
        assert self.CommonSetGene_run , '''Error: CommonSetGene has not been performed yet. You must run CommonSetGene and then try again.'''
      
    
        if SelGeneLoc.tolist()==False:
            self.SelGeneLoc = self.ExGeneLocList
        else:
            self.SelGeneLoc = SelGeneLoc
        
        UMAPFeatEmbedd = self.PredFeatEmbedd[self.SelGeneLoc]
            

        ## Performing UMAP to reduce the dimensionality of patient and gene vectors
        TotalFeatPred = np.concatenate([ UMAPFeatEmbedd, self.PredIndEmbedd, self.PredIndCentroid, self.PredFeatCentroid])

        TotaFeatUMAP = umap.UMAP(**UMAP_Parameters ) # n_neighbors=20, min_dist=1.0,  n_neighbors=30, min_dist=0.2
        TotaFeatUMAP.fit(TotalFeatPred)
        print(TotaFeatUMAP)
        TotaFeatUMAPProj = TotaFeatUMAP.transform(TotalFeatPred)

        self.PredFeatX = TotaFeatUMAPProj[: UMAPFeatEmbedd.shape[0], 0]
        self.PredFeatY = TotaFeatUMAPProj[:UMAPFeatEmbedd.shape[0], 1]

        self.PredIndX = TotaFeatUMAPProj[UMAPFeatEmbedd.shape[0]: (UMAPFeatEmbedd.shape[0] + self.PredIndEmbedd.shape[0]), 0]
        self.PredIndY = TotaFeatUMAPProj[UMAPFeatEmbedd.shape[0]: (UMAPFeatEmbedd.shape[0] + self.PredIndEmbedd.shape[0]), 1]
        
        
    # IndRepresent--------------------------------------------------------------------------------------------------------------------------------------------------------------------          
            
    def IndRepresent (self, mode='pooled',nrows=0,ncols=0):
        
        assert self.UMPA_run, '''Error: UMAP has not been performed yet. You must run UMAP(PerformUMPA()) and then try again.'''
        
        RefPatLongG = self.IndCentMembers[self.ReferencePatIDLong]
        RefPatShortG = 1- RefPatLongG
        IndRefThetaSub = self.IndRefTheta
        
        def PooledPlot ():
            
            if self.IndCentMembers[self.ReferencePatIDLong] == 0:
                palette = ['yellowgreen', 'red']
                LowRiskLabel=0
            else:
                palette = ['red', 'yellowgreen']
                LowRiskLabel=1

            plt.figure(figsize=(10,10))
            
            for i in range(self.NCL_Ind): #IndCentEvenMembers
                
                if i == LowRiskLabel:
                    IndRefThetaSub_ = np.argsort((1 -IndRefThetaSub),axis=0)
                    IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)
                else:
                    IndRefThetaSub_ = np.argsort(IndRefThetaSub,axis=0)
                    IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)
                        
                plt.scatter(self.PredIndX[self.IndCentMembers==i], self.PredIndY[self.IndCentMembers==i],  alpha=IndRefThetaSubDicho, marker='*', 
                            s =120, color=palette[i], edgecolors='black',linewidth=0.5)

            plt.scatter(self.PredIndX[self.ReferencePatIDShort], self.PredIndY[self.ReferencePatIDShort],  alpha=1, marker='P',  s =200, color=palette[RefPatShortG], edgecolors='black',linewidth=1.)    
            plt.scatter(self.PredIndX[self.ReferencePatIDLong], self.PredIndY[self.ReferencePatIDLong],  alpha=1, marker='P',  s =200, color=palette[RefPatLongG], edgecolors='black',linewidth=1.)    
            handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels

            CL1 = mlines.Line2D([], [], color=palette[RefPatLongG], marker='*', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Low-risk group')
            CL2 = mlines.Line2D([], [], color=palette[RefPatShortG], marker='*', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='High-risk group')
            CL3 = mlines.Line2D([], [], color=palette[RefPatLongG], marker='P', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Low-risk reference')
            CL4 = mlines.Line2D([], [], color=palette[RefPatShortG], marker='P', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='High-risk reference')

            handles.extend([CL1]+[CL2]+[CL3]+[CL4])
            #leg = plt.legend(handles=handles,prop={'size': 12})
            leg = plt.legend(handles=handles,prop={'size': 12}, loc='lower left',bbox_to_anchor=(0,-0.15), ncol=2)            
            
                        
        def ComprehPlot ():
            
            palette = np.array(sns.color_palette("hls", len(self.UniqueTumors)+1))[1:]
            if self.IndCentMembers[self.ReferencePatIDLong] == 0:
                markers = ['.', '*']
                LowRiskLabel=0
            else:
                markers = ['*','.']
                LowRiskLabel=1

            plt.figure(figsize=(10,10))

            AddLeg = []

            for idx, typeTumor in enumerate(self.UniqueTumors):
                IndCentMembersSub=  self.IndCentMembers[self.PostHocSet['tumor_type'] == typeTumor]
                PredIndXSub = self.PredIndX[self.PostHocSet['tumor_type'] == typeTumor]
                PredIndYSub = self.PredIndY[self.PostHocSet['tumor_type'] == typeTumor]
                IndRefThetaSub = self.IndRefTheta[self.PostHocSet['tumor_type'] == typeTumor]

                for i in range(self.NCL_Ind): #IndCentEvenMembers
                    
                    if i == LowRiskLabel:
                        IndRefThetaSub_ = np.argsort((1 -IndRefThetaSub),axis=0)
                        IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)
                    else:
                        IndRefThetaSub_ = np.argsort(IndRefThetaSub,axis=0)
                        IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)
                    
                    plt.scatter(PredIndXSub[IndCentMembersSub==i], PredIndYSub[IndCentMembersSub==i],  marker=markers[i], 
                                s =120, color=palette[idx], edgecolors='black',linewidth=1.2)

                AddLeg.append(mlines.Line2D([], [], color=palette[idx], marker='s', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label=typeTumor))
                
            plt.scatter(self.PredIndX[self.ReferencePatIDShort], self.PredIndY[self.ReferencePatIDShort],  alpha=0.7, marker='P',  s =250, color='red', edgecolors='black',linewidth=2.)    
            plt.scatter(self.PredIndX[self.ReferencePatIDLong], self.PredIndY[self.ReferencePatIDLong], alpha=0.7, marker='X',  s =200, color='yellowgreen', edgecolors='black',linewidth=2.)    

            handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
            CL1 = mlines.Line2D([], [], color='white', marker='.', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Low-risk group')
            CL2 = mlines.Line2D([], [], color='white', marker='*', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='High-risk group')
            CL3 = mlines.Line2D([], [], color='yellowgreen', marker='X', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Low-risk reference')
            CL4 = mlines.Line2D([], [], color='red', marker='P', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='High-risk reference')

            handles.extend(AddLeg+[CL1]+[CL2]+[CL3]+[CL4])
            #leg = plt.legend(handles=handles,prop={'size': 12}, loc='upper right')
            leg = plt.legend(handles=handles,prop={'size': 12}, loc='lower left',bbox_to_anchor=(0,-0.15), ncol=4)            

                       
        def EachPlot (nrows,ncols):
            
            assert nrows * ncols >= len(self.UniqueTumors), '''ERROR: The size arguments for subplot, norws (number of rows) and ncols (number of columns) were not passed correctly. 
                Please provide the correct nrows and ncols parameters in arument.'''
            
            if nrows % 2 !=0:
                nrows += 1
            if ncols % 2 !=0:
                ncols += 1
        
            if self.IndCentMembers[self.ReferencePatIDLong] == 0:
                palette = ['yellowgreen', 'red']
                LowRiskLabel=0
            else:
                palette = ['red', 'yellowgreen']
                LowRiskLabel=1

                
            plt.figure(figsize=(14,14))
            for idx, typeTumor in enumerate(self.UniqueTumors):
                
                IndCentMembersSub=  self.IndCentMembers[self.PostHocSet['tumor_type'] == typeTumor]
                PredIndXSub = self.PredIndX[self.PostHocSet['tumor_type'] == typeTumor]
                PredIndYSub = self.PredIndY[self.PostHocSet['tumor_type'] == typeTumor]
                IndRefThetaSub = self.IndRefTheta[self.PostHocSet['tumor_type'] == typeTumor]

                plt.subplot(nrows,ncols,idx+1)
                plt.subplots_adjust(hspace=0.15, wspace=0.12)
                for i in range(self.NCL_Ind): #IndCentEvenMembers
                    
                    if i == LowRiskLabel:
                        IndRefThetaSub_ = np.argsort((1 -IndRefThetaSub),axis=0)
                        IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)
                    else:
                        IndRefThetaSub_ = np.argsort(IndRefThetaSub,axis=0)
                        IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)
                        
                    plt.scatter(PredIndXSub[IndCentMembersSub==i], PredIndYSub[IndCentMembersSub==i],  alpha=IndRefThetaSubDicho, marker='*', 
                                s =90, color=palette[i], edgecolors='black',linewidth=0.5)
                plt.title(typeTumor, fontsize = 24)  

            handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
            CL1 = mlines.Line2D([], [], color=palette[RefPatLongG], marker='*', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Low-risk group')
            CL2 = mlines.Line2D([], [], color=palette[RefPatShortG], marker='*', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='High-risk group')

            handles.extend([CL1]+[CL2])
            plt.legend(handles=handles,prop={'size': 16}, loc='lower left',bbox_to_anchor=(-0.10,-0.2), ncol=ncols)
                
                
        if mode == 'pooled':
            PooledPlot()
        elif mode == 'each':
            EachPlot(nrows,ncols)
        elif mode == 'comprehensive':
            ComprehPlot()
        else:
            Warning.warn("Wrong mode value, only mode values 'pooled', 'each', and 'comprehensive' are allowed.")
            
            
    # IndGeneRepresent--------------------------------------------------------------------------------------------------------------------------------------------------------------------          
            
    def IndGeneRepresent (self, mode='whole',nrows=0,ncols=0 ):
        
        RefPatLongG = self.IndCentMembers[self.ReferencePatIDLong]
        RefPatShortG = 1- RefPatLongG
        
        self.ExFeatGLabel = np.argmax(self.PredFCosCLSim[self.SelGeneLoc], axis=1)
        paletteTumor = np.array(sns.color_palette("hls", len(self.UniqueTumors)+1))[1:]
        paletteUniqSet = np.array(sns.color_palette("hls", len(self.UniqSetGeneLoc)+1))
        ClPaletee = np.array(sns.color_palette("hls", self.NCL_Feat+1)) 
        
        
        def WholePlot ():
        
            if self.IndCentMembers[self.ReferencePatIDLong] == 0:
                markers = ['.', '*']
                palette = ['yellowgreen', 'red']
                LowRiskLabel=0
            else:
                markers = ['*','.']
                palette = ['red', 'yellowgreen']
                LowRiskLabel=1

            ## Plotting     
            plt.figure(figsize=(18,9))


            ## Comprehensive representation across cancer
            plt.subplot(121)
            plt.subplots_adjust(wspace=0.1)
            AddLeg = []
            for idx, typeTumor in enumerate(self.UniqueTumors):
                IndCentMembersSub=  self.IndCentMembers[self.PostHocSet['tumor_type'] == typeTumor]
                PredIndXSub = self.PredIndX[self.PostHocSet['tumor_type'] == typeTumor]
                PredIndYSub = self.PredIndY[self.PostHocSet['tumor_type'] == typeTumor]
                IndRefThetaSub = self.IndRefTheta[self.PostHocSet['tumor_type'] == typeTumor]
                 
                for i in range(self.NCL_Ind): #IndCentEvenMembers
                    
                    if i == LowRiskLabel:
                        IndRefThetaSub_ = np.argsort((1 -IndRefThetaSub),axis=0)
                        IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)
                    else:
                        IndRefThetaSub_ = np.argsort(IndRefThetaSub,axis=0)
                        IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)
                        
                    plt.scatter(PredIndXSub[IndCentMembersSub==i], PredIndYSub[IndCentMembersSub==i],  alpha=IndRefThetaSubDicho, marker=markers[i], 
                                s =120, color=paletteTumor[idx], edgecolors='black',linewidth=1.2)

                AddLeg.append(mlines.Line2D([], [], color=paletteTumor[idx], marker='s', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label=typeTumor))


            for i in np.unique(self.ExFeatGLabel):
                plt.scatter(self.PredFeatX[self.ExFeatGLabel==i], self.PredFeatY[self.ExFeatGLabel==i],  alpha=0.8, marker='p',  s =80, color=ClPaletee[i], edgecolors='black',linewidth=1.2)


            handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels

            CL1 = mlines.Line2D([], [], color='white', marker='.', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Low-risk group')
            CL2 = mlines.Line2D([], [], color='white', marker='*', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='High-risk group')
            CL5 = mlines.Line2D([], [], color='white', marker='p', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Genes')
            handles.extend(AddLeg+[CL1]+[CL2]+[CL5])
            #leg = plt.legend(handles=handles,prop={'size': 12}, loc='upper right')
            leg = plt.legend(handles=handles,prop={'size': 12}, loc='lower left',bbox_to_anchor=(0,-0.15), ncol=4)            
            plt.title('Comprehensive representation across cancer',fontsize = 15)
            
            AnnoLocAdjX = min(self.PredFeatX.min(),self.PredIndX.min()) - 1
            AnnoLocAdjY = min(self.PredFeatY.min(),self.PredIndY.min()) - 7.8

            plt.annotate('Selected cancer set: '+self.CommonSetKey + ',  N of genes: '+str(len(self.SelGeneLoc)),  xycoords='data', xy=(AnnoLocAdjX, AnnoLocAdjY),  
                         va = "bottom", ha="left",  fontsize=13,color='darkorange',
            bbox=dict(facecolor='none', edgecolor='lightgray', boxstyle='round,pad=0.5'),annotation_clip=False,xytext=(0, 0), textcoords='offset points')


            ## Risk-oriented representation across cancer
            IndRefThetaSub = self.IndRefTheta
            
            plt.subplot(122)
            for i in range(self.NCL_Ind): #IndCentEvenMembers
                
                if i == LowRiskLabel:
                    IndRefThetaSub_ = np.argsort((1 -IndRefThetaSub),axis=0)
                    IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)
                else:
                    IndRefThetaSub_ = np.argsort(IndRefThetaSub,axis=0)
                    IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)
                    
                plt.scatter(self.PredIndX[self.IndCentMembers==i], self.PredIndY[self.IndCentMembers==i],  alpha=IndRefThetaSubDicho, marker='*', 
                            s =120, color=palette[i], edgecolors='black',linewidth=0.5)


            for i in np.unique(self.ExFeatGLabel):
                plt.scatter(self.PredFeatX[self.ExFeatGLabel==i], self.PredFeatY[self.ExFeatGLabel==i],  alpha=0.8, marker='p',  s =80, color=ClPaletee[i], edgecolors='black',linewidth=1.2)

            plt.scatter(self.PredIndX[self.ReferencePatIDShort], self.PredIndY[self.ReferencePatIDShort],  alpha=0.7, marker='P',  s =250, color='red', edgecolors='black',linewidth=2.)    
            plt.scatter(self.PredIndX[self.ReferencePatIDLong], self.PredIndY[self.ReferencePatIDLong], alpha=0.7, marker='X',  s =200, color='yellowgreen', edgecolors='black',linewidth=2.)    
            handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
            CL1 = mlines.Line2D([], [], color='yellowgreen', marker='*', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Low-risk group')
            CL2 = mlines.Line2D([], [], color='red', marker='*', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='High-risk group')
            CL3 = mlines.Line2D([], [], color='yellowgreen', marker='X', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Low-risk reference')
            CL4 = mlines.Line2D([], [], color='red', marker='P', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='High-risk reference')
            CL5 = mlines.Line2D([], [], color='white', marker='p', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Genes')
            handles.extend([CL1]+[CL2]+[CL3]+[CL4]+[CL5])
            #leg = plt.legend(handles=handles,prop={'size': 12}, loc='upper right')
            leg = plt.legend(handles=handles,prop={'size': 12}, loc='lower left',bbox_to_anchor=(0.0,-0.15), ncol=3)            
            plt.title('Risk-oriented representation across cancer',fontsize = 15)
            
            
            
        def EachPlot ():

            if self.IndCentMembers[self.ReferencePatIDLong] == 0:
                palette = ['yellowgreen', 'red']
                LowRiskLabel=0
            else:
                palette = ['red', 'yellowgreen']
                LowRiskLabel=1


            plt.figure(figsize=(14,14))
            for idx, typeTumor in enumerate(self.UniqueTumors):

                IndCentMembersSub=  self.IndCentMembers[self.PostHocSet['tumor_type'] == typeTumor]
                PredIndXSub = self.PredIndX[self.PostHocSet['tumor_type'] == typeTumor]
                PredIndYSub = self.PredIndY[self.PostHocSet['tumor_type'] == typeTumor]
                IndRefThetaSub = self.IndRefTheta[self.PostHocSet['tumor_type'] == typeTumor]
                
                plt.subplot(nrows,ncols,idx+1)
                plt.subplots_adjust(hspace=0.15)

                for i in range(self.NCL_Ind): #IndCentEvenMembers
                    
                    if i == LowRiskLabel:
                        IndRefThetaSub_ = np.argsort((1 -IndRefThetaSub),axis=0)
                        IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)
                    else:
                        IndRefThetaSub_ = np.argsort(IndRefThetaSub,axis=0)
                        IndRefThetaSubDicho = (IndRefThetaSub_.argsort(axis=0)+1)/ (IndRefThetaSub_.shape[0]+1)                          
                    
                    plt.scatter(PredIndXSub[IndCentMembersSub==i], PredIndYSub[IndCentMembersSub==i],  alpha=IndRefThetaSubDicho, marker='*', 
                                s =90, color=palette[i], edgecolors='black',linewidth=0.5)
                for i in np.unique(self.ExFeatGLabel):
                    plt.scatter(self.PredFeatX[self.ExFeatGLabel==i], self.PredFeatY[self.ExFeatGLabel==i],  alpha=0.8, marker='p',  s =80, color=ClPaletee[i], edgecolors='black',linewidth=1.2)

                plt.title(typeTumor, fontsize = 15)  


            handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
            CL1 = mlines.Line2D([], [], color=palette[RefPatLongG], marker='*', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Low-risk group')
            CL2 = mlines.Line2D([], [], color=palette[RefPatShortG], marker='*', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='High-risk group')
            CL3 = mlines.Line2D([], [], color='white', marker='p', linestyle='None',  linewidth=1., markersize=12, markeredgewidth=1,markeredgecolor='black', label='Genes')


            handles.extend([CL1]+[CL2]+[CL3])
            plt.legend(handles=handles,prop={'size': 12}, loc='lower left',bbox_to_anchor=(-1.2,-0.15), ncol=4)            

            
        if mode == 'whole':
            WholePlot()
        elif mode == 'each':
            EachPlot()     
        else:
             Warning.warn("Wrong mode value, only mode values 'whole' and 'each' are allowed.")