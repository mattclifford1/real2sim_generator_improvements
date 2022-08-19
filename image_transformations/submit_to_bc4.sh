#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="64G"        # change these

job_name="trans"
time="0-12:00"
sbatch -t $time -J $job_name -o $dir$lr'.out' -e $dir$lr'.err' --mem=$ram server/submit_job.sh ./image_transformations/submit_to_bc4.sh
