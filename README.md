# Evaluating Representation Learning of Code Changes for Predicting Patch Correctness in Program Repair

```bibtex
@inproceedings{tian2020evaluating, 
  title={Evaluating Representation Learning of Code Changes for Predicting Patch Correctness in Program Repair}, 
  author={Tian, Haoye and Liu, Kui and Kabor{\'e}, Abdoul Kader and Koyuncu, Anil and Li, Li and Klein, Jacques and Bissyand{\'e}, Tegawend{\'e} F.},
  booktitle={Proceedings of the 35th IEEE/ACM International Conference on Automated Software Engineering}, 
  year={2020}, 
  publisher={ACM},
  url = {https://doi.org/10.1145/3324884.3416532}, 
  doi = {10.1145/3324884.3416532}
} 
```
Paper Link: https://ieeexplore.ieee.org/abstract/document/9286101

## Ⅰ) Catalogue of Repository

* ### data
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
* ### preprocess
  the preprocess of code file and data generation for RQ1 and RQ2.

* ### similarity_calculation
  the patch similarity statistics and filetra for RQ1 and RQ2.

* ### prediction 
  the classifier of patch correctness for RQ3.

## Ⅱ) Custom Prediction
To predict the correctness of your custom patches, you are welcome to use the prediction interface.

### A) Requirements for BERT
  * **BERT model client&server:** 24-layer, 1024-hidden, 16-heads, 340M parameters. download it [here](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip).
  * **Environment for BERT server** (different from reproduction)
    * python 3.7 
    * pip install tensorflow==1.14
    * pip install bert-serving-client==1.10.0
    * pip install bert-serving-server==1.10.0
    * pip install protobuf==3.20.1
    * Launch BERT server via `bert-serving-start -model_dir "Path2BertModel"/wwm_cased_L-24_H-1024_A-16 -num_worker=2 -max_seq_len=360`
  * **Patch snippet text:** your patch snippet.
### B) Predict
Let's give it a try!
```
python API.py predict $patch_text
```

