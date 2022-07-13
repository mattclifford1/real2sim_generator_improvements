#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these

dir="/user/work/mc15445/summer-project"
epochs=250
batch=64
lr="0.001"

# what pretrained model to use
model=/user/work/mc15445/summer-project/models/sim2real/alex/trained_gans/[surface_3d]/128x128_[shear]_250epochs/checkpoints/best_generator.pth
name=surface_shear

job_name="pre_no_gans"
time="0-5:00"
# train from scratch no no_gan
task="edge_2d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $dir --lr $lr --task $task --ram --pretrained_model $model --pretrained_name $name
task="edge_2d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $dir --lr $lr --task $task --ram --pretrained_model $model --pretrained_name $name
task="surface_3d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $dir --lr $lr --task $task --ram --pretrained_model $model --pretrained_name $name
task="surface_3d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $dir --lr $lr --task $task --ram --pretrained_model $model --pretrained_name $name

job_name="pre_gans"
time="0-7:00"
# train from scratch GAN
task="edge_2d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir $dir --lr $lr --task $task --ram --pretrained_model $model --pretrained_name $name
task="edge_2d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir $dir --lr $lr --task $task --ram --pretrained_model $model --pretrained_name $name
task="surface_3d tap"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir $dir --lr $lr --task $task --ram --pretrained_model $model --pretrained_name $name
task="surface_3d shear"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --GAN --batch_size $batch --epochs $epochs --dir $dir --lr $lr --task $task --ram --pretrained_model $model --pretrained_name $name
