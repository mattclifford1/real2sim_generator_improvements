#!/bin/bash
git pull

VENV=/user/work/mc15445/conda_envs/multi_task
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $VENV

python run_all.py --dir /user/work/mc15445/summer-project
