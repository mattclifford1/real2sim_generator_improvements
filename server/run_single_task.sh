#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these

data_dir="/user/work/mc15445/summer-project"
epochs=250
batch=64
lr="0.001"

job_name="no_gans"
time="0-5:00"
# train from scratch no no_gan
task="edge_2d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram
task="edge_2d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram
task="surface_3d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram
task="surface_3d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram

job_name="gans"
time="0-7:00"
# train from scratch GAN
task="edge_2d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram
task="edge_2d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram
task="surface_3d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram
task="surface_3d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram
