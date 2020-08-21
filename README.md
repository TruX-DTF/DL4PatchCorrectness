# Predicting Patch Correctness using Embeddings

## data
the dataset and results of each experiment.

* experiemnt1(train dataset)
  * Patches_train.zip: correct patches. the developer patches as committed in five open source project repositories.
  * APR-Efficiency-PFL: incorrect patches. the patches under the folders affixed with '\_C'.
* experiment2(test dataset)
  * The patches to be evaluated from RepairThemAll.
* experiment3
    * APR-Efficiency-NFL: the patches labeled with affix 'P' and '\_C', means 'palusible' and 'correct'.
	* DefectRepairing: the patches labeled with json file.
	* defects4j-developer: the correct patches.

## preprocess
preprocess of code file and data generation for RQ1 and RQ2.

## similarity_calculation
patch similarity statistics and filetra for RQ1 and RQ2.

## prediction 
classifier of patch correctness for RQ3.