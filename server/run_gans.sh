#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these
time="0-10:00"
job_name="gans"

epochs=300
batch=64
lr="0.001"

# train from scratch
task="edge_2d tap"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_gan.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="edge_2d shear"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_gan.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="surface_3d tap"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_gan.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
task="surface_3d shear"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_gan.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --task $task
