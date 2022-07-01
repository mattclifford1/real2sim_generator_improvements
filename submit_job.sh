#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=0:0:10
#SBATCH --mem=100M
#SBATCH --job-name=multi_task

# #SBATCH --gres=gpu:2
# cd /user/home/mc15445/summer-project/real2sim_multitask

module load languages/anaconda3/2021-3.9-bioconda
module load libs/cuda/10.2-gcc-5.4.0-2.26
module load tools/git/2.35.1

source /mnt/storage/software/languages/anaconda/anaconda.3.9-2021.12-bioconda/etc/profile.d/conda.sh
conda activate /user/work/mc15445/conda_envs/multi_task
echo 'Hello'
srun python test.py
srun python run_all.py --dir /user/work/mc15445/summer-project
wait
