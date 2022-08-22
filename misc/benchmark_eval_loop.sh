#!/bin/bash
####### TO CHANGE
VENV=sim2real # conda python virtual environment
NUM_JOBS=10         # how many parallel jobs to run

call_program(){
  # change in here to call the scipt that you want to time
  cd /home/matt/summer-project/real2sim_multitask
  # cd /home/matt/summer-project/real2sim_generator_improvements
  python image_transformations/eval_tranforms.py --dev --ram
}



# make sure conda is accessable
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $VENV

call_parallel(){
  for (( j=1; j<=$1; j++ ))
  do
  	call_program &
  done
  wait
  echo "==========================="
  echo "$1 jobs in parallel"
}

# Run loop
for (( i=1; i<=$NUM_JOBS; i++ ))
do
	time call_parallel $i
done

wait
echo "Finished"
