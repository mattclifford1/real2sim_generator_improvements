#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="64G"        # change these

job_name="trans"
time="0-12:00"
name="trans"
sbatch -t $time -J $job_name -o $dir$name'.out' -e $dir$name'.err' --mem=$ram server/submit_job.sh ./image_transformations/run_all_bc4.sh
