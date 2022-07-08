#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu

# SBATCH --time=0:0:30
# SBATCH --mem=64G
# SBATCH --job-name=multi_task
# SBATCH -e server/stderr.txt
# SBATCH -o server/stdout.txt

# #SBATCH --gres=gpu:2


# module load libs/cuda/10.2-gcc-5.4.0-2.26
module load languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch
# srun python run_all.py --dir /user/work/mc15445/summer-project

srun $*    # run all command line inputs
wait

# srun python trainers/train_unet.py --batch_size 256 --epochs 10
#  --dir /user/work/mc15445/summer-project
# wait
