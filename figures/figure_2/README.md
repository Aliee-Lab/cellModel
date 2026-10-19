# Code for reproducing Figure 2

This repository contains the code to reproduce **Figure 2** of the paper (PARM dataset). The figure first compares the performance of a universal (non-cell-type-specific) PARM model to the performance of cell-type-specific models, and then compares the distributions of model predictions to the ground truth.

In order to reproduce this figure, [PARM](https://github.com/vansteensellab/PARM) and the [PARM preprocessing pipeline](https://github.com/vansteensellab/PARM_preprocessing_pipeline) must be installed.

## Environment

All required packages are listed in `environment.yaml` and `requirements.txt`.

```bash
conda env create -f environment.yaml
conda activate <env_name>
# or, with pip:
pip install -r requirements.txt
```

PARM is installed as a regular (non-editable) package in this environment. The `patches/` directory in this repository (see below) contains the modified PARM source files needed to train a universal, multi-cell-type model; `scripts/sync_parm.sh` copies them into the installed package's `site-packages` location before training or prediction.

## Data

We use the MH4 library data of Barbadilla-Martinéz et al. (2026). The data can be accessed from the Gene Expression Omnibus with accession number [GSE301246](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE301246). We use the file `GSE301246_PARM_Training_Library_v2_MH4_raw_counts.txt.gz`.

To train our universal PARM model, we processed the data with the [PARM preprocessing pipeline](https://github.com/vansteensellab/PARM_preprocessing_pipeline) using the regulatory features data in the `example_data` directory in the pipeline repository and the configuration file `all_cell_type_config.yml`. Note that this gives the same folds as those used to train the original, cell-type-specific PARM models.

During the analysis, we use the original metadata files to visualize the ground-truth distributions of activity levels. These metadata files 

## Repository structure

```
.
├── environment.yml
├── requirements.txt
├── patches/                              # modified PARM source files for the universal model
│   ├── PARM_utils_load_model.py          #   linear1 sized to n_cell_types instead of 1
│   ├── PARM_train.py                     #   cell_line threading, NaN-masked loss, per-celltype stats
│   ├── PARM_predict.py                   #   multi-column predictions, per-celltype test-fold evaluation
│   └── __main__.py                       #   CLI support for multi-celltype --cell_type
├── scripts/
│   └── sync_parm.sh                      # copies patches/ into the installed PARM package
├── train_universal_model.sh              # sbatch script: 5-fold CV training of the universal model
└── PARM_universal_vs_celltype_specific.ipynb   # analysis + Figure 2 plotting
```


## Reproducing Figure 2

1. **Preprocess the data.** Run the PARM preprocessing pipeline on the raw counts using `all_cell_type_config.yml`, whose `CELLTYPES_TO_COLUMNS` entry lists all 7 cell types, producing five fold-level HDF5 files (`fold0.hdf5` … `fold4.hdf5`), each containing all 7 cell types' `Log2RPM_*` columns.

2. **Patch PARM for multi-output training.** Run `scripts/sync_parm.sh` (edit the `EDITED_PARM_DIR` and `PARM_SITE_PKG` paths at the top first) to copy the modified files in `patches/` into the active conda environment's installed PARM package.

3. **Train the universal model.** Run `train_universal_model.sh`, which calls `parm train` once per fold (5-fold leave-one-out, as in the original PARM training), passing all 7 cell type names jointly via `--cell_type "A__B__C__D__E__F__G"`. This trains a single shared convolutional backbone with one output node per cell type, per fold.

4. **Obtain cell-type-specific baseline predictions.** We compare against the original, pretrained cell-type-specific PARM models from the [PARM repository](https://github.com/vansteensellab/PARM) directly, rather than retraining them.

5. **Generate predictions on the held-out test folds** for both the universal model and the pretrained cell-type-specific models, using `parm predict --predict_test_fold`. For the universal model, pass the same joined `--cell_type` string used at training time so output columns are labeled correctly.

6. **Compare performance and prediction distributions.** Open `PARM_universal_vs_celltype_specific.ipynb`, which reads both sets of test-fold predictions and reproduces Figure 2's performance comparison and prediction/ground-truth distribution plots.