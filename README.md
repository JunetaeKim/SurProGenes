# SurProGenes

  This repository provides the Python codes to implement the experiments in
  [_**SurProGenes: Survival Risk-Ordered Representation of Cancer Patients and Genes for the Identification of Prognostic Genes**_](https://proceedings.mlr.press/v202/kim23s.html).
  This paper suggests a framework that recommends individual prognostic genes by integrating representation learning and statistical analysis. The study has been accepted as a poster presentation at [ICML 2023](https://icml.cc/Conferences/2023/Dates). The authors are currently working on the camera-ready version of the paper and plan to provide a link to the full text in the near future.


## Model
  We propose a collaborative filtering-derived mechanism to represent patients in order of their survival risk and dichotomize them. The model consists of three novel mechanisms. The code for the main model is provided in [`RCFR_AC.py`](./2.Analysis/Models/RCFR_AC.py). See Section 3 for a detailed explanation.
  <p align="center">
  <img src="./2.Analysis/Models/Model_figures/fig_model_outline.png" width="600"/>
  </p>

### Similarity-based Embedding Mechanism (SEM)
  <p align="center">
  <img src="./2.Analysis/Models/Model_figures/fig_sem_mechanism.png" width="500"/>
  <em>SEM</em>
  </p>

  $$loss_{GX}=\frac{1}{PG}\sum_{p=1}^{P}{\sum_{g=1}^{G}{(GX_{p,g}-\widehat{WGX}_{p,g})^2}}$$

### Risk-Ordering Mechanism (ROM)

1. Patient-oriented ROM (PROM)
  <p align="center">
  <img src="./2.Analysis/Models/Model_figures/fig_prom.png" width="400"/>
  <em>PROM</em>
  </p>

  $$Rloss_{pat}=\frac{1}{P^2}\sum_{i=1}^{P}{\sum_{j=1}^{P}{PRM_{i,j}(DTL_{i,j}-PED_{i,j})^2}}$$

2. Gene-oriented ROM (GROM)
  <p align="center">
  <img src="./2.Analysis/Models/Model_figures/fig_grom.png" alt width="500"/>
  <em>GROM</em>
  </p>
  <p align="center">
  <img src="./2.Analysis/Models/Model_figures/fig_acam.png" width="400"/>
  <em>ACAM</em>
  </p>
  
  $$Rloss_{gene}=\frac{1}{G^2}\sum_{i=1}^{G}{\sum_{j=1}^{G}{(GLDeud_{i,j}-GED_{i,j})^2}}$$

### Dichotomization & Clustering Mechanism

  $$CHloss_{gene}=\frac{1}{G}\sum_{g=1}^{G}{\max{(D_{\theta}(GEM_{g},GCM)-\xi_{1},0)}}$$

  $$CHloss_{pat}=\frac{1}{P}\sum_{p=1}^{P}{\max{(D_{\theta}(PEM_{p},PCM)-\xi_{2},0)}}$$
  
  $$SPloss_{gene}=\frac{1}{C^G}\sum_{i=1}^{C^G}{\max{(\xi_{3}-(D_{\theta}^{s}(GCM))_{i},0)}}$$
  
  $$SPloss_{pat}=\frac{1}{C^P}\sum_{i=1}^{C^P}{\max{(\xi_{3}-(D_{\theta}^{s}(PCM))_{i},0)}}$$
 

## Prerequisites

  The code is implemented with the following dependencies:

  * python (v3.8)
  * numpy (v1.19)
  * tensorflow (v2.4)
  * jupyter
  * lifelines
  * matplotlib
  * pandas
  * scikit-learn
  * seaborn
  * statsmodels
  * venn


## Implementation the experiments

  1. Clone the repository.
  
  2. Create a conda environment with all necessary libraries in **Prerequisites**.

#### Data processing

  3. Run [`1.Data/1.DataProcessing_Raw.ipynb`](./1.Data/1.DataProcessing_Raw.ipynb) in Jupyter notebook.
  
  4. Run [`1.Data/2.DataProcessing_CoreProgenex_GroupNorm.ipynb`](./1.Data/2.DataProcessing_CoreProgenex_GroupNorm.ipynb) in Jupyter notebook. Then, you can get the following data in [`1.Data/ProcessedData/`](./1.Data/ProcessedData/) folder.
  
  - `StakedgData_GroupNorm.npy`
  - `GeneToInt_GroupNorm.npy`
  - `IntToGene_GroupNorm.npy`
  - `DisimInd_GroupNorm.npy`
  - `TTE_GroupNorm.npy`
  - `Event_GroupNorm.npy`
  - `LogAnalData.pickle`
  
#### Model training
  
  5. Open terminal and activate your conda environment. Then, run the training files in [`2.Analysis/`](./2.Analysis/) as follows:
  
  ```
  python 1.1.Train_CFR.py
  python 1.2.Train_RCFR_NoGROM.py
  python 1.3.Train_RCFR.py
  python 1.4.Train_RCFR_AC_NoCL.py
  python 1.5.Train_RCFR_AC_NoSim.py
  python 1.6.Train_RCFR_AC_W1.py
  python 1.7.Train_RCFR_AC_W2.py
  python 1.8.Train_RCFR_AC_W3.py
  ```
  
#### Evaluation
  
  6. Run the model selection files in [`2.Analysis/`](./2.Analysis/) as follows:
  
  ```
  python 2.1.Select_BestCFR.py -n 100
  python 2.2.Select_BestRCFR_NoGROM.py -n 100
  python 2.3.Select_BestRCFR.py -n 100
  python 2.4.Select_BestRCFR_AC_NoCL.py -w W3 -n 100
  python 2.5.Select_BestRCFR_AC_Nosim.py -w W3 -n 100
  python 2.6.Select_BestRCFR_AC.py -m M06 -w W1 -n 100
  python 2.6.Select_BestRCFR_AC.py -m M07 -w W2 -n 100
  python 2.6.Select_BestRCFR_AC.py -m M08 -w W3 -n 100
  ```
  
  7. Run [`2.Analysis/3.1.Performance_Comparison.ipynb`](./2.Analysis/3.1.Performance_Comparison.ipynb) in Juypter notebook. Then, you can get evaluation results including [`Performance Table`](./2.Analysis/EvalResults/PerformanceTable.csv). For readers interested only in statistical analysis, skipping the time-consuming model training and best model selection procedures, the authors have uploaded the results for both procedures in [`2.Analysis/EvalResults`](./2.Analysis/EvalResults).



#### Additional experiments

  8. For *performance comparison by candidate size*, run model selection files in [`2.Analysis/`](./2.Analysis/) as follows:

  ```
  python 2.1.Select_BestCFR.py -n 300
  python 2.1.Select_BestCFR.py -n 500
  python 2.1.Select_BestCFR.py -n 700
  python 2.1.Select_BestCFR.py -n 900
  python 2.2.Select_BestRCFR_NoGROM.py -n 300
  python 2.2.Select_BestRCFR_NoGROM.py -n 500
  python 2.2.Select_BestRCFR_NoGROM.py -n 700
  python 2.2.Select_BestRCFR_NoGROM.py -n 900
  python 2.3.Select_BestRCFR.py -n 300
  python 2.3.Select_BestRCFR.py -n 500
  python 2.3.Select_BestRCFR.py -n 700
  python 2.3.Select_BestRCFR.py -n 900
  python 2.4.Select_BestRCFR_AC_NoCL.py -w W3 -n 300
  python 2.4.Select_BestRCFR_AC_NoCL.py -w W3 -n 500
  python 2.4.Select_BestRCFR_AC_NoCL.py -w W3 -n 700
  python 2.4.Select_BestRCFR_AC_NoCL.py -w W3 -n 900
  python 2.5.Select_BestRCFR_AC_Nosim.py -w W3 -n 300
  python 2.5.Select_BestRCFR_AC_Nosim.py -w W3 -n 500
  python 2.5.Select_BestRCFR_AC_Nosim.py -w W3 -n 700
  python 2.5.Select_BestRCFR_AC_Nosim.py -w W3 -n 900
  python 2.6.Select_BestRCFR_AC.py -m M06 -w W1 -n 300
  python 2.6.Select_BestRCFR_AC.py -m M06 -w W1 -n 500
  python 2.6.Select_BestRCFR_AC.py -m M06 -w W1 -n 700
  python 2.6.Select_BestRCFR_AC.py -m M06 -w W1 -n 900
  python 2.6.Select_BestRCFR_AC.py -m M07 -w W2 -n 300
  python 2.6.Select_BestRCFR_AC.py -m M07 -w W2 -n 500
  python 2.6.Select_BestRCFR_AC.py -m M07 -w W2 -n 700
  python 2.6.Select_BestRCFR_AC.py -m M07 -w W2 -n 900
  python 2.6.Select_BestRCFR_AC.py -m M08 -w W3 -n 300
  python 2.6.Select_BestRCFR_AC.py -m M08 -w W3 -n 500
  python 2.6.Select_BestRCFR_AC.py -m M08 -w W3 -n 700
  python 2.6.Select_BestRCFR_AC.py -m M08 -w W3 -n 900
  ```
  
  9. Run [`2.Analysis/3.2.Performance_Comparison_by_Candidate_Size.ipynb`](./2.Analysis/3.2.Performance_Comparison_by_Candidate_Size.ipynb) in Jupyter notebook.

  10. For *statistical analisys with the main model*, run [`2.Analysis/4.1.BestRCFR_AC_PostEval.ipynb`](./2.Analysis/4.1.BestRCFR_AC_PostEval.ipynb) in Jupyter notebook. For readers interested only in statistical analysis, skipping the time-consuming model training and best model selection procedures, the authors have uploaded the results for both procedures in [`2.Analysis/ModelResults`](./2.Analysis/ModelResults).



## Notice
#### Your Contributions to Enhancing Our Gene Analysis Algorithm
We appreciate your interest in our gene expression algorithm and its potential for dichotomizing patients based on survival risk while simultaneously recommending genomes that create significant differences between the dichotomized groups. We recognize that our preliminary method, particularly our Similarity-based Embedding Mechanism (SEM), may not fully account for certain genes with similar expression levels in both groups. However, we perceive this not merely as a limitation but rather as a valuable opportunity to further refine and enhance our approach.

We want to emphasize that our Patient-Oriented Risk Ordering Mechanism (PROM) has demonstrated robustness and reliability, reinforcing our confidence in the overall value of our work. We are committed to improving our methodology to more effectively address the intricacies of gene expression scenarios, and we have a firm belief in the bright future of this research.

We wholeheartedly welcome your constructive feedback and suggestions. Please do not hesitate to share your insights with us. By working together, we can fine-tune this promising algorithm.
