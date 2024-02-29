#!/bin/sh

#SBATCH --job-name=modified_echo  # Job name
#SBATCH --output=output/modified_echo-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=output/modified_echo-%A.err  # Standard error of the script
#SBATCH --time=0-24:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 12 per GPU)
#SBATCH --mem=40G  # Memory in GB (Don't use more than 48G per GPU unless you absolutely need it and know what you are doing)

# load python module
source ~/miniconda3/etc/profile.d/conda.sh

# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate diff_us

# run the program
# 256
#python semantic_diffusion_model/image_sample.py --datadir /home/data/farid/2CH_ED_augmented/ --dataset_mode camus_full_2CH --resume_checkpoint output/b-maps/dataset_camus-b_map_min_1.0-img_size_256_lr_0.0001/2024-02-26/model030000.pt --results_dir output/results --num_samples 15 --is_train False --inference_on_train True --img_size 256 --num_channels 256 --batch_size_test 10 --batch_size_train 10 --distributed_data_parallel False
# 128
#python semantic_diffusion_model/image_sample.py --datadir /home/data/farid/2CH_ED_augmented/ --resume_checkpoint output/b-maps/dataset_camus-b_map_min_1.0-img_size_128_lr_1e-05/2024-02-26/ema_0.9_050000.pt --results_dir output/results --num_samples 15 --is_train False --inference_on_train True --img_size 128 --num_channels 128 --batch_size_test 10 --batch_size_train 10 --distributed_data_parallel False

#python semantic_diffusion_model/image_sample.py --datadir /home/data/farid/camus/augmented_camus/ --dataset_mode camus_full_2CH --resume_checkpoint output/b-maps/dataset_camus_full_2CH-b_map_min_0.95-img_size_128_lr_1e-05/2024-02-27/ema_0.99_024000.pt --results_dir output/results --num_samples 15 --is_train False --inference_on_train True --img_size 128 --num_channels 128 --batch_size_test 10 --batch_size_train 10 --distributed_data_parallel False
#camus
#python semantic_diffusion_model/image_sample.py --datadir /home/data/farid/camus/augmented_camus/ --dataset_mode camus_full_2CH --resume_checkpoint output/b-maps/dataset_camus_full_2CH-b_map_min_0.98-img_size_128-lr_1e-05-diffusion_steps_4000/2024-02-27/model050000.pt --results_dir output/results --num_samples 60 --is_train False --inference_on_train True --img_size 128 --num_channels 128 --batch_size_test 10 --batch_size_train 10 --distributed_data_parallel False --diffusion_steps 4000 --b_map_min 0.98 --num_workers 10
#thyroid
python semantic_diffusion_model/image_sample.py --datadir /home/data/farid/THYROID_MULTILABEL_2D_3D/imagesTrain/2D/ --dataset_mode thyroid --resume_checkpoint output/b-maps/dataset_thyroid-b_map_min_0.97-img_size_128-lr_1e-05-diffusion_steps_4000/2024-02-28/model032000.pt --results_dir output/results --num_samples 60 --is_train False --inference_on_train True --img_size 128 --num_channels 128 --batch_size_test 10 --batch_size_train 10 --distributed_data_parallel False --diffusion_steps 4000 --b_map_min 0.97 --num_workers 10
