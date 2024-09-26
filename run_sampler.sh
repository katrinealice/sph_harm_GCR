#!/bin/bash

#SBATCH --time 4:00:00
#SBATCH --mem 64G
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 50
#SBATCH --job-name standard
#SBATCH --array=1 #-40 #Run 40 copies of the code = 4000 samples
#SBATCH --partition cosma8-serial
#SBATCH --account dp270
#SBATCH --output /cosma8/data/dp270/dc-glas1/slurm-out/slurm-%A_%a.out

source ~/.bashrc
conda deactivate
conda activate hera_sim

SCRIPT="/cosma/home/dp270/dc-glas1/GCR_sampler/diffuse_emission/vis_realified.py"

echo $@

export OMP_NUM_THREADS=1 
python -u $SCRIPT "$@" -dir=standard \
                       -nsamples=200 \
                       -data_seed=20 \
                       -prior_seed=30 \
                       -lmax=20 \
                       -NLST=10 \
                       -lst_start=0.\
                       -lst_end=8.\
                       -dish_dia=14.6\
                       -zero_S_inv=false \
                       -cosmic_var=false \
                       -jobid=$SLURM_ARRAY_TASK_ID
