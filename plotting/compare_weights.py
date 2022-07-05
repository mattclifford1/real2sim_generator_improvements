import os
from argparse import ArgumentParser
from gan_tester import get_weights
import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd

def get_results(ARGS, gen=('edge_2d','tap'), dis=('edge_2d','tap')):
    gen_model_dir = os.path.join(ARGS.dir, 'models/sim2real/alex/trained_gans/['+gen[0]+']/128x128_['+gen[1]+']_250epochs')
    discrim_model_dir = os.path.join(ARGS.dir, 'models/sim2real/alex/trained_gans/['+dis[0]+']/128x128_['+dis[1]+']_250epochs')
    return get_weights(gen_model_dir, discrim_model_dir)



if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    parser.add_argument("--dev",  default=False, action='store_true', help='only run for a few samples when in dev mode')
    ARGS = parser.parse_args()

    # get all combinations of models
    task = ['edge_2d','surface_3d']
    sampling = ['tap', 'shear']
    generators = list(itertools.product(task, sampling))
    discrims = list(itertools.product(task, sampling))

    for generator in tqdm(generators, desc="Generators", leave=False):
        for discrim in tqdm(discrims, desc="Discriminators", leave=False):
            weights, layers = get_results(ARGS, generator, discrim)
