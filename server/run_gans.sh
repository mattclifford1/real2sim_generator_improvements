#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these
time="0-16:00"
job_name="no gans"

epochs=500
batch=64
lr="0.001"

# train from scratch no no_gan
task="edge_2d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="edge_2d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="surface_3d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="surface_3d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task

job_name="gans"
# train from scratch GAN
task="edge_2d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="edge_2d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="surface_3d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="surface_3d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
