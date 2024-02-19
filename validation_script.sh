#!/bin/bash

#SBATCH --job-name=CAMUS_INFERENCE
#SBATCH --output=CAMUS_INF-%A.out
#SBATCH --error=CAMUS_INF-%A.err
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --nodelist=dagobah
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
~/.conda/envs/camus/bin/python3 semantic_diffusion_model/image_sample.py --resume_checkpoint output/b-maps/.pt --results_dir output/results/b-maps --num_samples 10 --is_train False --inference_on_train True

ml -cuda -miniconda3