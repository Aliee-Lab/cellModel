# Code for reproducing Figure 3

This repository contains the code to reproduce **Figure 3** of the paper (RPE1 dataset). The figure compares two methods, **STATE** and **GEARS**, which are run separately.

## Environment

All required packages are listed in `environment.yml` and `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate <env_name>
# or, with pip:
pip install -r requirements.txt
```

## Data

We use the Replogle RPE1 CRISPRi Perturb-seq dataset (Replogle et al., 2022), accessed via the Hugging Face repository `arcinstitute/Replogle-Nadig-Preprint`.

To download it:

```bash
pip install huggingface_hub
huggingface-cli download arcinstitute/Replogle-Nadig-Preprint --repo-type dataset --local-dir <data_dir>
```

Set `<data_dir>` to wherever you want the data stored, and point the scripts and notebooks to this path.

## Repository structure

```
.
├── environment.yml
├── requirements.txt
├── state_pipeline_full/                 # STATE step 1: run the pipeline
├── RPE1_state_reproducibility.ipynb     # STATE step 2: analyse the output
└── RPE1_gears_reproducibility.ipynb     # GEARS: run directly
```

## Reproducing Figure 3

STATE and GEARS are run independently.

### STATE (pipeline, then notebook)

1. Run the pipeline in `state_pipeline_full/` to generate the results:
```bash
   cd state_pipeline_full
   python <script_name>.py <args>
   cd ..
```
   The output is saved to `<output_path>`.

2. Open `RPE1_state_reproducibility.ipynb` and run all cells to analyse the output and produce the STATE part of the figure:
```bash
   jupyter notebook RPE1_state_reproducibility.ipynb
```

### GEARS (notebook only)

GEARS runs directly from the notebook, with no separate script. Open `RPE1_gears_reproducibility.ipynb` and run all cells:

```bash
jupyter notebook RPE1_gears_reproducibility.ipynb
```
