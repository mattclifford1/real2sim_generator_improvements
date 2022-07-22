#!/bin/bash
dir='matt'
scp -r mc15445@bc4login.acrc.bris.ac.uk:~/storage/summer-project/models/sim2real/$dir ~/Downloads/sim2real


# to copy on bc4 all of the training csvs but not models/images
# $ find edge_2d/ -name "*.csv" -exec cp --parents \{\} ~/storage/temp/ \;

# to resume a partially downloaded scp use eg:   (only tested on a single zip file not using -r for a whole folder)
# rsync -P -rsh=ssh mc15445@bc4login.acrc.bris.ac.uk:/user/home/mc15445/storage/summer-project/models/sim2real/sim2real_diff_data_size.zip /home/matt/Downloads/
