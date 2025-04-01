#!/bin/sh
cd /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouzhejian-240108120128/Mol/e3_diffusion_for_molecules
source activate
conda activate py310
python finetune.py --config_path ./outputs/edm_qm9 \
    --reward xTB \
    --name edm_DDPO_5e-7 \
    --schedular DDPM \
    --num_train_timesteps 1000 \
    --train_clip_range 0.2 \
    --train_learning_rate 0.0000005 