# Predicting Patch Correctness using Embeddings

## data
the dataset and results of each experiment

1. experiemnt1(train dataset)
the patches used for inferring similarity threshold.

* Patches_train.zip: correct patches. the developer patches as committed in five open source project repositories.
* APR-Efficiency-PFL: incorrect patches. the patches under the folders affixed with '\_C'.

2. experiment2(test dataset)
the patches to be evaluated from RepairThemAll.

3. experiment3
the train and test dataset for prediction of patch correctness based on embedding.

* APR-Efficiency-NFL: the patches labeled with affix 'P' and '\_C', means 'palusible' and 'correct'.
* DefectRepairing: the patches labeled with json file.
* defects4j-developer: the correct patches.

## preprocess
preprocess of code file and data generation for RQ1 and RQ2.

## similarity_calculation:
patch similarity statistics and filetra for RQ1 and RQ2.

## prediction 
classifier of patch correctness for RQ3.