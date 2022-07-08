#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these


# parameter search to train models from scratch
time="0-2:30"    # accordingly
job_name="MT"
epochs=100
lr="0.01"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size 256 --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr
lr="0.001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size 256 --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr
lr="0.0001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size 256 --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr
lr="0.00001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size 256 --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr
lr="0.000001"

# test out multiple gpu job
ram="64G"
lr="0.0001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job_2.sh python trainers/train_unet.py --batch_size 512 --multi_GPU --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr

# parameter search to train model from checkpoint
ram="64G"
time="0-2:30"    # accordingly
job_name="MT-pre"
epochs=100
lr="0.0001"
model=/user/work/mc15445/summer-project/models/sim2real/alex/trained_gans/[surface_3d]/128x128_[shear]_250epochs/checkpoints/best_generator.pth
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size 256 --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --pretrained_model $model
lr="0.00001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size 256 --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --pretrained_model $model
lr="0.000001"
