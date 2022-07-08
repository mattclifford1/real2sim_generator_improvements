#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these


# parameter search to train models from scratch
time="0-7:00"
job_name="MT"
epochs=250
batch=256
lr="0.01"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr

batch=64
lr="0.001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr
batch=128
lr="0.001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr
batch=256
lr="0.001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr
# test out multiple gpu job
ram="64G"
lr="0.0001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job_2.sh python trainers/train_unet.py --batch_size $batch --multi_GPU --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr

# parameter search to train model from checkpoint
ram="32G"
job_name="MT-pre"
batch=128

model=/user/work/mc15445/summer-project/models/sim2real/alex/trained_gans/[surface_3d]/128x128_[shear]_250epochs/checkpoints/best_generator.pth
name=surface_shear
lr="0.001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --pretrained_model $model --pretrained_name $name

model=/user/work/mc15445/summer-project/models/sim2real/alex/trained_gans/[surface_3d]/128x128_[tap]_250epochs/checkpoints/best_generator.pth
name=surface_tap
lr="0.001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --pretrained_model $model --pretrained_name $name

model=/user/work/mc15445/summer-project/models/sim2real/alex/trained_gans/[edge_2d]/128x128_[tap]_250epochs/checkpoints/best_generator.pth
name=edge_tap
lr="0.001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --pretrained_model $model --pretrained_name $name

model=/user/work/mc15445/summer-project/models/sim2real/alex/trained_gans/[edge_2d]/128x128_[shear]_250epochs/checkpoints/best_generator.pth
name=edge_shear
lr="0.001"
sbatch -t $time -J $job_name$lr -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_unet.py --batch_size $batch --epochs $epochs --dir /user/work/mc15445/summer-project --lr $lr --pretrained_model $model --pretrained_name $name
