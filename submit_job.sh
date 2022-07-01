#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=0:0:10
#SBATCH --mem=100M
#SBATCH --job-name=multi_task

# #SBATCH --gres=gpu:2
# cd /user/home/mc15445/summer-project/real2sim_multitask
srun conda activate /user/work/mc15445/conda_envs/multi_task
srun python run_all.py --dir /user/work/mc15445/summer-project
wait
