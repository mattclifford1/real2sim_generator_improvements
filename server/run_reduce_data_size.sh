#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these

data_dir="/user/work/mc15445/summer-project"
epochs=250
batch=64
lr="0.0002"
task="surface_3d shear"

job_name="no_gans"
time="0-6:00"
# train from scratch no no_gan
data_size="1"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --eval_downstream
data_size="0.9"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --eval_downstream
time="0-5:00"
data_size="0.75"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --eval_downstream
time="0-4:00"
data_size="0.5"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --eval_downstream
time="0-3:00"
data_size="0.25"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --eval_downstream
time="0-3:00"
data_size="0.1"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --eval_downstream
data_size="0.05"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --eval_downstream
data_size="0.01"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --eval_downstream

# what pretrained model to use
model=/user/work/mc15445/summer-project/models/sim2real/alex/trained_gans/[edge_2d]/128x128_[tap]_250epochs/checkpoints/best_generator.pth
name=edge_tap

job_name="pre_no_gans"
time="0-6:00"
# train from scratch no no_gan
data_size="1"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --pretrained_model $model --pretrained_name $name --eval_downstream
data_size="0.9"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --pretrained_model $model --pretrained_name $name --eval_downstream
time="0-5:00"
data_size="0.75"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --pretrained_model $model --pretrained_name $name --eval_downstream
time="0-4:00"
data_size="0.5"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --pretrained_model $model --pretrained_name $name --eval_downstream
time="0-3:00"
data_size="0.25"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --pretrained_model $model --pretrained_name $name --eval_downstream
time="0-3:00"
data_size="0.1"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --pretrained_model $model --pretrained_name $name --eval_downstream
data_size="0.05"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --pretrained_model $model --pretrained_name $name --eval_downstream
data_size="0.01"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh python trainers/train_single_task.py --batch_size $batch --epochs $epochs --dir $data_dir --lr $lr --task $task --ram --data_size $data_size --pretrained_model $model --pretrained_name $name --eval_downstream
