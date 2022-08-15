# Experiments that need running still

PLOT:
- GAN vs noGAN
  - from scratch
  - transfer
  - RUN: GAN on its own?



IQA metrics:
- image transformations vs metrics/downstream
  - run for small data size
  - run for whole dataset (think how to optimise this for batch processing?)
    - batch size is all values of the transform?
    - set up as arg to input what transform to use (then can run on multiple nodes)
    - save results to csv file















DONE ===========================================

MISC:
- number of parameters in gen/disc (for computational analysis)


TODO: code:
- get average and std of training/validation runs (need to plot std) run more runs!
- plot of MAE for val set PoseNet
  - real
  - sim
