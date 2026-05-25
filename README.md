# Exploring HRMS-Compatible Molecular Fingerprints for Prediction of Granular Activated Carbon Removal of Organic Contaminants
 
**Greta Tödtmann** | Stockholm University
 
This repository contains the datasets, code, and trained models associated with my master's thesis. The work investigates the use of high-resolution mass spectrometry (HRMS)-compatible molecular fingerprints as features for machine learning models that predict the removal efficiency of organic contaminants by granular activated carbon (GAC).
 
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
BV10 regressor and RE classifier models trained on molecular fingerprints and DOC data.
 
### `residualized_models/`
DOC-residualized BV10 regressor and RE classifier fingerprint models used for feature importance analysis.
 
### `multitask_models/`
Multitask learning implementations that jointly predict RE and BV10.
