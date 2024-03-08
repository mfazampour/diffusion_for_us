#!/bin/sh

#SBATCH --job-name=modified_echo  # Job name
#SBATCH --output=output/modified_echo-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=output/modified_echo-%A.err  # Standard error of the script
#SBATCH --time=0-24:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 12 per GPU)
#SBATCH --mem=48G  # Memory in GB (Don't use more than 48G per GPU unless you absolutely need it and know what you are doing)

# load python module
source ~/miniconda3/etc/profile.d/conda.sh

# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate diff_us

# run the program
# for 256
#python semantic_diffusion_model/image_train.py --datadir /home/data/farid/camus/augmented_camus/ --dataset_mode camus_full_2CH --savedir output/b-maps --batch_size_train 4 --is_train True --random_flip False --deterministic_train False \
#        --use_fp16 True --distributed_data_parallel True --img_size 256 --lr_anneal_steps 50000 --num_channels 256 --b_map_min 0.98 --fp16_scale_growth 4e-3  \
#        --resume_checkpoint output/b-maps/dataset_camus-b_map_min_1.0-img_size_256_lr_0.0001/2024-02-25/model025000.pt
# for 128
#python semantic_diffusion_model/image_train.py --datadir /home/data/farid/camus/augmented_camus/ --dataset_mode camus_full_2CH --savedir output/b-maps --batch_size_train 4 --is_train True \
#        --random_flip False --deterministic_train False --use_fp16 True --distributed_data_parallel True --img_size 128 --lr_anneal_steps 100000 --num_channels 128 \
#        --b_map_min 0.98 --save_interval 4000 --lr 1e-5 --fp16_scale_growth 4e-3 --diffusion_steps 4000 \
#        --resume_checkpoint output/b-maps/dataset_camus_full_2CH-b_map_min_0.98-img_size_128-lr_1e-05-diffusion_steps_4000/2024-02-27/model050000.pt
# --drop_rate 0.0

# thyroid dataset
python semantic_diffusion_model/image_train.py --datadir /home/data/farid/THYROID_MULTILABEL_2D_3D/imagesTrain/2D/ --dataset_mode thyroid --savedir output/b-maps --batch_size_train 4 --is_train True \
        --random_flip False --deterministic_train False --use_fp16 True --distributed_data_parallel True --img_size 128 --lr_anneal_steps 100000 --num_channels 128 \
        --b_map_min 0.97 --save_interval 1000 --lr 1e-5 --fp16_scale_growth 4e-3 --diffusion_steps 1000 \
#         --resume_checkpoint output/b-maps/dataset_thyroid-b_map_min_1.0-img_size_128-lr_1e-05-diffusion_steps_4000/2024-02-28/model004000.pt

# liver dataset
#python semantic_diffusion_model/image_train.py --datadir /home/data/farid/simulated_images_cs_Demir_Yichen_Daniel/ --dataset_mode liver --savedir output/b-maps --batch_size_train 4 --is_train True \
#        --random_flip False --deterministic_train False --use_fp16 True --distributed_data_parallel True --img_size 256 --lr_anneal_steps 100000 --num_channels 256 \
#        --b_map_min 0.97 --save_interval 2000 --lr 1e-5 --fp16_scale_growth 4e-3 --diffusion_steps 2000 \
#         --resume_checkpoint output/b-maps/dataset_thyroid-b_map_min_1.0-img_size_128-lr_1e-05-diffusion_steps_4000/2024-02-28/model004000.pt
