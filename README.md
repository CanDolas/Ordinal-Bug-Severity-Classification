# Ordinal-Bug-Severity-Classification
Code and data for an empirical study comparing ordinal (HvL) and nominal (OvO, OvR) decomposition strategies for bug priority and severity prediction.
## Data Preparation

Download the raw dataset from the replication package of 
   Acharya & Ginde (2024):
   https://zenodo.org/records/10892319
   (file: `Mozilla_complete_dataset.csv.csv`)

Place the downloaded file, in the same 
   directory as `preprocessing_v1.ipynb`.

Run `preprocessing_v1.ipynb` to generate the leakage-free 
   fold files under `folds_output/`. All random operations 
   (fold splitting, SMOTE, undersampling) use a fixed seed (42), 
   so the generated folds are fully reproducible.
