#!/bin/bash
#SBATCH --partition=h200
#SBATCH --qos=h200_prio
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1 
#SBATCH --time=24:00:00 
#SBATCH -o /Users/kendri01/logs/multi-train.out 
#SBATCH -e /Users/kendri01/logs/multi-train.err 
#SBATCH -J parm_multi_train 

condaBin=/Users/kendri01/miniforge3/bin
source ${condaBin}/activate parm_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"


cd /mnt/scratche/slow/halab/Jack/parm

parm train --check_cuda

# Fold 0 model
parm train \
  --input PARM_preprocessing_pipeline/all_cell_type_out_data/onehot/tss_selection_m300_p100/fold[1234].* \
  --validation PARM_preprocessing_pipeline/all_cell_type_out_data/onehot/tss_selection_m300_p100/fold0.hdf5 \
  --output all_cell_type_fold0 \
  --cell_type HepG2__LNCaP__K562__HCT116__MCF7__U2OS__HEK293

# Fold 1 model
parm train \
  --input PARM_preprocessing_pipeline/all_cell_type_out_data/onehot/tss_selection_m300_p100/fold[0234].* \
  --validation PARM_preprocessing_pipeline/all_cell_type_out_data/onehot/tss_selection_m300_p100/fold1.hdf5 \
  --output all_cell_type_fold1 \
  --cell_type HepG2__LNCaP__K562__HCT116__MCF7__U2OS__HEK293

# Fold 2 model
parm train \
  --input PARM_preprocessing_pipeline/all_cell_type_out_data/onehot/tss_selection_m300_p100/fold[0134].* \
  --validation PARM_preprocessing_pipeline/all_cell_type_out_data/onehot/tss_selection_m300_p100/fold2.hdf5 \
  --output all_cell_type_fold2 \
  --cell_type HepG2__LNCaP__K562__HCT116__MCF7__U2OS__HEK293

# Fold 3 model
parm train \
  --input PARM_preprocessing_pipeline/all_cell_type_out_data/onehot/tss_selection_m300_p100/fold[0124].* \
  --validation PARM_preprocessing_pipeline/all_cell_type_out_data/onehot/tss_selection_m300_p100/fold3.hdf5 \
  --output all_cell_type_fold3 \
  --cell_type HepG2__LNCaP__K562__HCT116__MCF7__U2OS__HEK293

# Fold 4 model
parm train \
  --input PARM_preprocessing_pipeline/all_cell_type_out_data/onehot/tss_selection_m300_p100/fold[0123].* \
  --validation PARM_preprocessing_pipeline/all_cell_type_out_data/onehot/tss_selection_m300_p100/fold4.hdf5 \
  --output all_cell_type_fold4 \
  --cell_type HepG2__LNCaP__K562__HCT116__MCF7__U2OS__HEK293
