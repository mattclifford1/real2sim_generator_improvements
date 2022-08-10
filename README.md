# real2sim Generator Experiments
Collection of tools and experiments on the generator in the sim2real paper (LINK HERE).

## Baseline performance
Test generators on other tasks:

## Comparison of models
Weights of all the generators
(TODO: make mean/std of a load of weights of loads of models)

## Training the generator
Scripts to train the generator with varying options inside [trainers](trainers). eg:
```
$ python xxxx
```
TODO: write docs for this
### Training on the server (blue crystal)
Scipts inside [server](server).
TODO: write readmes

## Image transformation effect
Look at the readme inside [image_transformations](image_transformations). From here you can runt he user interface as well as quantitative experiments 





# Testing
Make tests inside the [pytest](pytest) folder and to run all test use:
```
$ pytest
```
This requires the pytest module to be downloaded eg with pip:
```
$ pip install pytest
```
