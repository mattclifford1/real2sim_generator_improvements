Adapatation of Alex Church's sim2real tactip repo. Code is designed to be lighter on dependancies (won't require the tactile_gym package nor other robotics libraries (vsp etc.))

# Running
use pix2pix.py to train the GAN eg:
```
$ python tactile_gym_sim2real/pix2pix.py --dir .. --task edge_2d tap
```

# Running on server (BC4)
Run the script `run_bc4.sh`
