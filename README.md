# Unraveling Renal Cell Carcinoma Subtypes and Prognoses by Integrative Histopathology-Genomics Analysis

This repository contains the trained convolutional neural networks for predicting renal cell carcinoma subtypes, molecular profiles, and prognoses.

All models are trained using Talos with keras and tensorflow-gpu backend. 

## Tumor identification
Under the tumor_normal/ directory. Our approaches achieved areas under the receiver operating characteristic curve (AUC) in the independent validation cohort of 0.964-0.985.

## Subtype classification
Under the subtypes/ directory. Our models attained an accuracy of 0.935 in the independent validation set.

## Overall survival prediction
Under the survival/ directory. Our prediction models distinguished the longer-term survivors from shorter-term survivors among stage I clear cell renal cell carcinoma (ccRCC) patients (log-rank test p = 0.02).

## Mutation status prediction
Under the mutations/ directory. We showed that morphological patterns weakly predicted somatic mutation profiles in the three major subtypes of renal cell carcinoma.

## Copy number alteration prediction
Under the copy_number_alteration/ directory. We predicted the copy number alterations in ccRCC patients in multiple genes (e.g., VHL, EGFR, KRAS, and WT1) with AUC > 0.7.

## Tumor mutation burden prediction
Under the tumor_mutation_burden/ directory. We demonstrated that the histopathology-based features are moderately correlated with tumor mutation burden measured by whole-exome sequencing (Spearman's correlation 0.419; correlation test p = 0.0003).


## Instructions on using talos
The full documentation of talos could be found [here](https://autonomio.github.io/talos/#/).

### Quick installation
```
pip install talos
```

### Usage and examples
[A short example](https://nbviewer.jupyter.org/github/autonomio/talos/blob/master/examples/Hyperparameter%20Optimization%20on%20Keras%20with%20Breast%20Cancer%20Data.ipynb).

[A comprehensive example](https://nbviewer.jupyter.org/github/autonomio/talos/blob/master/examples/Hyperparameter%20Optimization%20with%20Keras%20for%20the%20Iris%20Prediction.ipynb).
