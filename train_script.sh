#!/bin/bash

#SBATCH --job-name=modified_echo
#SBATCH --output=modified_echo-%A.out
#SBATCH --error=modified_echo-%A.err
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --nodelist=kessel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marina.dominguez@tum.de

# Load necessary modules
ml cuda
ml miniconda3

# Directly source the Conda environment activation script
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate camus

# Print the Python path for debugging
which python3

# Use the full path to Python from the 'camus' environment
# Run your Python script
# ~/.conda/envs/camus/bin/python3 semantic_diffusion_model/image_train.py --datadir augmented_camus/2CH_ED_augmented/ --savedir output/dm_saved_models/replica --batch_size_train 12 --is_train True --save_interval 50000 --lr_anneal_steps 50000 --random_flip True --deterministic_train False --use_fp16 True --distributed_data_parallel False --img_size 256
~/.conda/envs/camus/bin/python3 semantic_diffusion_model/image_train.py --datadir augmented_camus/2CH_ED_augmented/ --savedir output/b-maps --batch_size_train 12 --is_train True --random_flip False --deterministic_train False --use_fp16 True --distributed_data_parallel True --img_size 256 --lr_anneal_steps 50000

ml -cuda -miniconda3

