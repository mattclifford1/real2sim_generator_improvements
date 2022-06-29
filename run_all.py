import os
from argparse import ArgumentParser
from gan_tester import run
import numpy as np
import itertools
from tqdm import tqdm

def get_results(ARGS, model=('edge_2d','tap'), data=('edge_2d', 'tap')):
    gan_model_dir = os.path.join(ARGS.dir, 'models/sim2real/alex/trained_gans/['+model[0]+']/128x128_['+model[1]+']_250epochs')
    real_images_dir = os.path.join(ARGS.dir, 'data/Bourne/tactip/real/'+model[0]+'/'+model[1]+'/csv_val/images')
    sim_images_dir = os.path.join(ARGS.dir, 'data/Bourne/tactip/sim/'+model[0]+'/'+model[1]+'/128x128/csv_val/images')
    return run(gan_model_dir, real_images_dir, sim_images_dir, ARGS.dev)




if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    parser.add_argument("--dev",  default=False, action='store_true', help='only run for a few samples when in dev mode')
    ARGS = parser.parse_args()

    # get all combinations of models and data
    task = ['edge_2d','surface_3d']
    sampling = ['tap', 'shear']
    generators = list(itertools.product(task, sampling))
    discrims = list(itertools.product(task, sampling))
    datas = list(itertools.product(task, sampling))

    for generator in tqdm(generators, desc="Generators", leave=False):
        for data in tqdm(datas, desc="Data", leave=False):
            MSEs, discrim_scores = get_results(ARGS, generator, data)
            # print('Gen:', generator, ' Data: ', data)
            # print(np.mean(MSEs))
            # print(np.mean(discrim_scores))
            # print('=====================')
