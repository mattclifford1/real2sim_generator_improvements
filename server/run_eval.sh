#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these

data_dir="/user/work/mc15445/summer-project"
task="surface_3d shear"
batch_size=32

job_name="eval"
time="0-1:30"

# run=0
# sbatch -t $time -J $job_name$run -o $dir$run'.out' -e $dir$run'.err' --mem=$ram server/submit_job.sh python validation/val_from_list.py --batch_size $batch --dir $data_dir --task $task --run $run
# run=1
# sbatch -t $time -J $job_name$run -o $dir$run'.out' -e $dir$run'.err' --mem=$ram server/submit_job.sh python validation/val_from_list.py --batch_size $batch --dir $data_dir --task $task --run $run
run=2
sbatch -t $time -J $job_name$run -o $dir$run'.out' -e $dir$run'.err' --mem=$ram server/submit_job.sh python validation/val_from_list.py --batch_size $batch_size --dir $data_dir --task $task --run $run
