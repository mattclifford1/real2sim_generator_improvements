#!/bin/bash
dir='matt'
scp -r mc15445@bc4login.acrc.bris.ac.uk:~/storage/summer-project/models/sim2real/$dir ~/Downloads


# to copy on bc4 all of the training csvs but not models/images
# $ find edge_2d/ -name "*.csv" -exec cp --parents \{\} ~/storage/temp/ \;
