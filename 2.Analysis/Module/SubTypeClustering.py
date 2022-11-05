import numpy as np
import pandas as pd
import warnings
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist


#---------------------------------------------------------------------------------------------------------------------------###
class SubTyping:
    
    def __init__ (self, BaseDataset, ApplDataset, SelGeneLoc,IntToGene,  ApplScaling = 1, KM_pat=2, KM_gene=2,  metric = 'chebyshev',
                  method = 'ward' , cmap = 'vlag', standard_scale = 1, row_cluster=True, col_cluster=True):
        
        assert 'Risk' in ApplDataset.columns, 'No Risk variable in the ApplDataset, Risk must be generated befor inputing the dataset'
        
        ApplDataset['RiskDesc'] =(np.round(ApplDataset['Risk'], 1)*10).astype('int32')
        ApplDataset = ApplDataset.sort_values('Risk')
        
        self.BaseDataset = BaseDataset
        self.ApplDataset = ApplDataset
        self.SelGeneLoc = SelGeneLoc
        self.IntToGene = IntToGene
        self.KM_pat = KM_pat
        self.KM_gene = KM_gene
        self.ApplScaling = ApplScaling
        self.metric = metric
        self.method = method
        self.cmap = cmap
        self.standard_scale = standard_scale
        self.row_cluster = row_cluster
        self.col_cluster = col_cluster
             
    
    def LocLabelingOrig (self, Z, mk):

        Mem = fcluster(Z, mk, 'maxclust')
        
        Palette =  sns.color_palette(self.CLColParName, mk)
        Palette =  [Palette[i-1] for i in Mem]

        return Mem, Palette   
    

    def LocLabelingAppl (self, Z, re_index, mk):

        Mem = fcluster(Z, mk, 'maxclust')
        Palette =  sns.color_palette(self.CLColParName, mk)
        Palette =  [Palette[i-1] for i in Mem[re_index]]

        return Mem[re_index], Palette
    
    
    def OrderingData (self, UnOrdData, OrdCol, OrdRow):
        OrdTumorType =  None
        OrdRiskDesc = None
        OrdRisk = None
        
        #Ordeirng Dataset over patients
        try:
            OrdDataset = UnOrdData.iloc[OrdCol].copy()
        except:
            print('''WARNING: Linkage and ordering information has not be obtained, which may result in errors''')
            OrdDataset = UnOrdData.copy()
            
        if 'tumor_type' in UnOrdData.columns:
            OrdTumorType = OrdDataset['tumor_type'].values  
        if 'RiskDesc' in UnOrdData.columns:
            OrdRiskDesc = OrdDataset['RiskDesc'].values  
        if 'Risk' in UnOrdData.columns:
            OrdRisk = OrdDataset['Risk'].values  

        # Ordeirng Dataset over Genes
        SeqResGene = [self.IntToGene[self.SelGeneLoc[i]+1]  for i in OrdRow]
        OrdDataset = OrdDataset[SeqResGene].copy()    
        
        return OrdDataset, OrdTumorType, OrdRiskDesc, OrdRisk
    
        
    

    def VisualInfo (self, OrdGeneMem, OrdPatMem, OrdTumorType,  RiskDesc, TumorType ):
        
        # Color palette for rows and columns of heatmaps
        RiskColPar = np.array([self.RiskColPar[i] for i in RiskDesc])
        TumorColPar = np.array([self.TumorColPar[i] for i in np.unique(TumorType, return_inverse=True)[1] ])
        
        # Masking vectors
        PatGene2D = OrdGeneMem[:,None] * OrdPatMem[None]
        PatGeneMask1 = OrdGeneMem[:,None]  == PatGene2D / OrdGeneMem[:,None]
        PatGeneMask2 = OrdPatMem[None]  == PatGene2D / OrdPatMem[None]
        PatGene2DMask = PatGeneMask1 * PatGeneMask2
        

        # Loc information for highlight
        HighlightLoc = []
        for i in range(1,self.KM_pat+1):
            for j in range(1,self.KM_gene+1):
                col = np.where(OrdPatMem==i)[0].min()
                row = np.where(OrdGeneMem==j)[0].min()
                w = np.sum((OrdPatMem==i))
                h = np.sum((OrdGeneMem==j))
                HighlightLoc.append([col, row, w, h])
        HighlightLoc = np.array(HighlightLoc)  

        
        # Dectecting for tumor sequence start point and length
        PrevTumor = ''
        StartLoc = 0
        StartLocList =[]
        TumorSeqList =[]

        for num, i in enumerate (OrdTumorType):    
            if PrevTumor != i :
                StartLocList.append(StartLoc)
                PrevTumor = i
                TumorSeqList.append(i)
            StartLoc += 1
        StartLocList = np.array(StartLocList + [OrdTumorType.shape[0]])
        TumorSeqList = np.array(TumorSeqList)
        TumLocList = np.concatenate([StartLocList[:-1, None] , StartLocList[1:, None] - StartLocList[:-1, None]], axis=-1)

        # Dectecting for tumor sequence start point of row and column and lenght of row and column
        ExtTumLocList = []

        for SubLoc in HighlightLoc:

            SelLoc = (TumLocList[:, 0] >= SubLoc[0]) & (TumLocList[:, 0] <  SubLoc[0] + SubLoc[2])
            SubTumLoc = pd.DataFrame(TumLocList[SelLoc], columns=['col','w'])
            SubTumLoc['row'] = SubLoc[1]
            SubTumLoc['h'] = SubLoc[3]
            SubTumLoc = SubTumLoc[['col','row','w','h']].values

            ExtTumLocList.append(SubTumLoc)
        ExtTumLocList = np.concatenate(ExtTumLocList)
        
        
        # Nested clustering labeling
        NestedPatCL = np.concatenate([ np.ones(i) *(num+1)  for num, i in enumerate(TumLocList[:, 1])])

        # Color palette for tumors
        TumorSeqColPar = np.array([ TumorColPar[i] for i in np.unique(TumorSeqList, return_inverse=True)[1] ])
        
        return PatGene2DMask, HighlightLoc, TumLocList, ExtTumLocList, NestedPatCL, TumorSeqColPar, RiskColPar, TumorColPar
        
    
    
    

    def ClusterBase(self, RiskColPar, TumorColPar, CLColParName="Accent"):
        
        self.RiskColPar = RiskColPar
        self.TumorColPar = TumorColPar
        self.CLColParName = CLColParName
        
        # Conduct initial clustering to obtain linkage results, ordered results of patients and genes, and etc. 
        self.HeatRes = sns.clustermap((self.BaseDataset.T), row_cluster=self.row_cluster, col_cluster= self.col_cluster, cmap= self.cmap,  
                                 standard_scale=self.standard_scale, metric=self.metric, method=self.method)
        plt.close()

        
        self.GeneMemOrig, self.FGPaletteOrig = self.LocLabelingOrig(self.HeatRes.dendrogram_row.calculated_linkage, self.KM_gene)
        self.PatMemOrig, self.PGPaletteOrig = self.LocLabelingOrig(self.HeatRes.dendrogram_col.calculated_linkage, self.KM_pat)

        GeneMemOrig_sort = self.GeneMemOrig[self.HeatRes.dendrogram_row.reordered_ind]
        PatMemOrig_sort = self.PatMemOrig[self.HeatRes.dendrogram_col.reordered_ind]
        
        
        # Ordeirng ApplDataset over patients and genes OrderingData (self, UnOrdData, OrdCol, OrdRow)
        OrdCol = self.HeatRes.dendrogram_col.reordered_ind 
        OrdRow = self.HeatRes.dendrogram_row.reordered_ind
        self.OrdApplDataset, self.OrdApplTumorType, self.OrdApplRiskDesc, self.OrdApplRisk = self.OrderingData(UnOrdData=self.ApplDataset, OrdCol=OrdCol, OrdRow=OrdRow)
        
    
        # Obtaining information for visualization
        VisualReturnBase = self.VisualInfo(GeneMemOrig_sort, PatMemOrig_sort, self.OrdApplTumorType,  self.ApplDataset['RiskDesc'].values, self.ApplDataset['tumor_type'])
        self.PatGene2DMaskBase, self.HighlightLocBase, self.TumLocListBase, self.ExtTumLocListBase, self.NestedPatCLBase, \
        self.TumorSeqColParBase,self.RiskColParBase, self.TumorColParBase = VisualReturnBase
        
        
        
        
    def ClusterAppl(self,):
        
        self.RiskColParAppl = np.array([self.RiskColPar[i] for i in self.OrdApplRiskDesc])
        self.TumorColParAppl = np.array([ self.TumorColPar[i] for i in np.unique(self.OrdApplTumorType, return_inverse=True)[1] ])
        
        self.GeneMemAppl, self.FGPaletteAppl = self.LocLabelingAppl(self.HeatRes.dendrogram_row.calculated_linkage, self.HeatRes.dendrogram_row.reordered_ind, self.KM_gene)
        self.PatMemAppl, self.PGPaletteAppl = self.LocLabelingAppl(self.HeatRes.dendrogram_col.calculated_linkage, self.HeatRes.dendrogram_col.reordered_ind, self.KM_pat)
        
        

    def ClusterUniversal(self,):
        

        self.HeatResUniv = sns.clustermap(self.OrdApplDataset.T, row_cluster=self.row_cluster, col_cluster= self.col_cluster, 
                                     cmap=self.cmap, standard_scale=self.standard_scale, metric=self.metric, method=self.method)
        
        
        plt.close()
        self.GeneMemOrigUniv, self.FGPaletteOrigUniv = self.LocLabelingOrig(self.HeatResUniv.dendrogram_row.calculated_linkage, self.KM_gene)
        self.PatMemOrigUniv, self.PGPaletteOrigUniv = self.LocLabelingOrig(self.HeatResUniv.dendrogram_col.calculated_linkage, self.KM_pat)
        

        GeneMemOrigUniv_sort = self.GeneMemOrigUniv[self.HeatResUniv.dendrogram_row.reordered_ind]
        PatMemOrigUniv_sort = self.PatMemOrigUniv[self.HeatResUniv.dendrogram_col.reordered_ind]

        self.OrdUnivTumorType = self.OrdApplTumorType[self.HeatResUniv.dendrogram_col.reordered_ind]
        # Obtaining information for visualization
        VisualReturnUniversal = self.VisualInfo(GeneMemOrigUniv_sort, PatMemOrigUniv_sort, self.OrdUnivTumorType,  self.OrdApplRiskDesc, self.OrdApplTumorType )
        self.PatGene2DMaskUniv, self.HighlightLocUniv, self.TumLocListUniv, self.ExtTumLocListUniv, self.NestedPatCLUniv, \
        self.TumorSeqColParUniv,self.RiskColParUniv, self.TumorColParUniv = VisualReturnUniversal
        
        
        
    
    def Eval_CL (self, Dataset, SelGeneLoc,IntToGene, 
                          standard_scale = 1, row_cluster=True, col_cluster=True ,
                          List_KM_pat=[3,4,5,6,7,8,9,10], 
                          List_KM_gene=[3,4,5,6,7,8,9,10], 
                          List_metric = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',  'cosine', 'euclidean', 'mahalanobis',  'minkowski', 'seuclidean'],
                          List_method = [ 'complete', 'average', 'weighted', 'centroid',  'median','ward'] ):

        SelGeneNames = [IntToGene[i+1] for i in SelGeneLoc]
        DropList = Dataset.columns[~Dataset.columns.isin(SelGeneNames)]
        DatasetEXGene = Dataset.drop(columns=DropList)


        TabCols = ['KM_pat','KM_gene','Metric','Method',  'SH_Risk', 'MeanRD_W']
        MetricValues = pd.DataFrame(columns=TabCols)


        for kp in List_KM_pat:
            for kg in List_KM_gene:
                for mrc in List_metric:
                    for mth in List_method:
                        
                        #try:
                        
                            # Conduct initial clustering to obtain linkage results, ordered results of patients and genes, and etc. 
                            HeatEval = sns.clustermap((DatasetEXGene.T), row_cluster=row_cluster, col_cluster= col_cluster, 
                                                      standard_scale=standard_scale, metric=mrc, method=mth)
                            plt.close() 

                            # Ordeirng ApplDataset over patients and genes OrderingData 
                            OrdCol = HeatEval.dendrogram_col.reordered_ind
                            OrdRow = HeatEval.dendrogram_row.reordered_ind
                            OrdEvalEXSet, OrdTumorTypeEval, OrdRiskDescEval,OrdRiskEval  = self.OrderingData(Dataset, OrdCol=OrdCol, OrdRow=OrdRow)

                            # Obtaining clustering label
                            PatMemEval, PGPaletteEval = self.LocLabelingAppl(HeatEval.dendrogram_col.calculated_linkage, HeatEval.dendrogram_col.reordered_ind, kp)
                            
                            if len(np.unique(PatMemEval))  ==1:
                                print('error info:', kp, kg, mrc, mth, )
                                continue

                            # Sub clustering labeling                            
                            OrdTumorType = np.unique(OrdTumorTypeEval, return_inverse=True)[1]
                            PatMemTumCL = np.core.defchararray.add(PatMemEval.astype(str), OrdTumorType.astype(str) ) #PatMemEval:member, OrdTumorType:type
                            PatMemTumCLEncod = np.unique(PatMemTumCL, return_inverse=True)[1]


                            # Distance matrix
                            DistMat_ExpRisk = cdist(OrdRiskEval[:,None], OrdRiskEval[:,None], metric=mrc)


                            # silhouette_score for risk
                            SH_Risk = silhouette_score(DistMat_ExpRisk, PatMemEval, metric = 'precomputed')


                            # Distance among risks within a sub cluster
                            AvgDistMat_SubRisk = []
                            for i in np.unique(PatMemTumCLEncod):
                                SubOrdRisk =  OrdRiskEval[PatMemTumCLEncod == i]

                                if SubOrdRisk.shape[0] > 1:
                                    DistMat_SubRisk = cdist(SubOrdRisk[:,None], SubOrdRisk[:,None], metric=mrc)
                                    numerator = np.sum(DistMat_SubRisk)
                                    denominator =(DistMat_SubRisk.shape[0]**2 - DistMat_SubRisk.shape[0])/2

                                    AvgDistMat_SubRisk.append(numerator / denominator)

                            MeanRD_W = np.mean(AvgDistMat_SubRisk)




                            print(kp, kg, mrc, mth,  ', SH_Risk: ', SH_Risk,  ', MeanRD_W: ', MeanRD_W)

                            SubMetricValues = pd.DataFrame([[kp, kg, mrc, mth, SH_Risk,MeanRD_W]], columns=TabCols)
                            MetricValues = pd.concat([MetricValues,SubMetricValues], axis=0)

                        
                       # except KeyboardInterrupt:
                       #     plt.close()
                       #     os.sys.exit(0)
                       # except:
                       #     plt.close()
                       #     pass
                            

        return MetricValues
    
    

