'''
make a data store locally of the models held on disk to easy access and reading of models
'''
import os
from argparse import ArgumentParser
import pandas as pd
import sys; sys.path.append('..'); sys.path.append('.')

def get_all_models_in_dir(dir, model_name='best_generator.pth'):
    models = []
    for dirpath, _, filenames in os.walk(dir):
        for f in filenames:
            models.append(os.path.relpath(os.path.join(dirpath, f), dir))
    return models








if __name__ == '__main__':
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default=os.path.join('..', 'models', 'sim2real', 'matt'), help='path to folder models are held')
    ARGS = parser.parse_args()

    models = get_all_models_in_dir(ARGS.dir)
    
