#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these
time="0-8:00"

batch=64
lr="0.001"
# train from scratch
task="edge_2d tap"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="edge_2d shear"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="surface_3d tap"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="surface_3d shear"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task


# train from model checkpoint
model=/user/work/mc15445/summer-project/models/sim2real/matt/edge_2d/tap/not_pretrained/LR\:0.001_decay\:0.1_BS\:64/models/231.pth
name=edge_2d_tap_simple

task="edge_2d shear"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --pretrained_model $model --pretrained_name $name --task $task
task="surface_3d tap"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --pretrained_model $model --pretrained_name $name --task $task
task="surface_3d shear"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --pretrained_model $model --pretrained_name $name --task $task
