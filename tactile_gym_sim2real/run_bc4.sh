#!/bin/bash
dir="tactile_gym_sim2real/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these

data_dir="/user/work/mc15445/summer-project"
epochs=250
batch=64
lr="0.001"


job_name="alex"
time="0-7:00"
# train from scratch GAN
task="edge_2d tap"
name='et'
sbatch -t $time -J $name$job_name -o $dir$name'.out' -e $dir$name'.err' --mem=$ram server/submit_job.sh python tactile_gym_sim2real/pix2pix.py --epochs $epochs --dir $data_dir
task="edge_2d shear"
name='es'
sbatch -t $time -J $name$job_name -o $dir$name'.out' -e $dir$name'.err' --mem=$ram server/submit_job.sh python tactile_gym_sim2real/pix2pix.py --epochs $epochs --dir $data_dir
task="surface_3d tap"
name='st'
sbatch -t $time -J $name$job_name -o $dir$name'.out' -e $dir$name'.err' --mem=$ram server/submit_job.sh python tactile_gym_sim2real/pix2pix.py --epochs $epochs --dir $data_dir
task="surface_3d shear"
name='ss'
sbatch -t $time -J $name$job_name -o $dir$name'.out' -e $dir$name'.err' --mem=$ram server/submit_job.sh python tactile_gym_sim2real/pix2pix.py --epochs $epochs --dir $data_dir
