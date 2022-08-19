#!/bin/bash

git pull
# to copy on bc4 all of the training csvs but not models/images
cd ~/storage/summer-project/models/sim2real/matt/
$ find * -name "*.csv" -exec cp --parents \{\} ~/summer-project/real2sim_multitask/gan_models/training_csvs/ \;
git add .
git commit -m 'bc4 upload of training stats'
git push
