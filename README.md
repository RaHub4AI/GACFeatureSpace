# Exploring HRMS-Compatible Molecular Fingerprints for Prediction of Granular Activated Carbon Removal of Organic Contaminants
 
**Greta Tödtmann** | Stockholm University
 
This repository contains the datasets and code associated with my master's thesis. The work investigates the use of high-resolution mass spectrometry (HRMS)-compatible molecular fingerprints as features for machine learning models that predict the removal efficiency of organic contaminants by granular activated carbon (GAC).
 
---
 
## Repository Structure
 
```
├── data/                   # Processed datasets used for model training and evaluation
├── mapping/                # UMAP workflow and PubChemLite subset used for chemspace mapping
├── raw_models/             # raw fingerprint + DOC models
├── residualized_models/    # DOC-residualized models
└── multitask_models/       # Multitask learning models trained on both BV10 and RE data
```
 
### `data/`
Contains cleaned BV10 regression set and RE classification set, as well as the cleaning workflow used to produce these datasets.
 
### `mapping/`
Contains the workflow to generate chemical space mappings of datasets onto a subset of PubChemLite.
 
### `raw_models/`
Contains BV10 regressor and RE classifier workflows for training on molecular fingerprints and DOC data.
 
### `residualized_models/`
Contains DOC-residualized fingerprint BV10 regressor and RE classifier workflows used for feature importance analysis.
 
### `multitask_models/`
Contains multitask learning implementations that jointly predict RE and BV10.
