# Evaluating Representation Learning of Code Changes for Predicting Patch Correctness in Program Repair

```bibtex
@inproceedings{tian2020evaluating, 
  title={Evaluating Representation Learning of Code Changes for Predicting Patch Correctness in Program Repair}, 
  author={Tian, Haoye and Liu, Kui and Kabor{\'e}, Abdoul Kader and Koyuncu, Anil and Li, Li and Klein, Jacques and Bissyand{\'e}, Tegawend{\'e} F.},
  booktitle={Proceedings of the 35th IEEE/ACM International Conference on Automated Software Engineering}, 
  year={2020}, 
  publisher={ACM}
} 
```

## data
the dataset and results of each experiment.

* experiemnt1
  * Patches_train.zip: the developer patches as committed in five open source project repositories.
  * APR-Efficiency-PFL: the patches under the folders affixed with '\_C'.
* experiment2
  * The patches to be evaluated from RepairThemAll.
* experiment3
    * APR-Efficiency-NFL: the patches labeled with affix '\_P' and '\_C', means 'palusible' and 'correct'.
	* DefectRepairing: the patches labeled with json file.
	* defects4j-developer: the correct patches.

## preprocess
preprocess of code file and data generation for RQ1 and RQ2.

## similarity_calculation
patch similarity statistics and filetra for RQ1 and RQ2.

## prediction 
classifier of patch correctness for RQ3.
