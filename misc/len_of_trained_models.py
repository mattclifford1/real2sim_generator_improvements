import os
from argparse import ArgumentParser
import pandas as pd
import sys; sys.path.append('..'); sys.path.append('.')

def get_all_models_in_dir(dir, csv_file='training_stats.csv', expected_len=251):
    for dirpath, _, filenames in os.walk(dir):
        for f in filenames:
            if f == csv_file:
                path = os.path.join(dirpath, f)
                df_len = len(pd.read_csv(path))
                if df_len != expected_len:
                    print(path)
                    print(len(pd.read_csv(path)))
                    print('')
            # models.append(os.path.relpath(os.path.join(dirpath, f), dir))








if __name__ == '__main__':
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default=os.path.join('..', 'models', 'sim2real', 'matt'), help='path to folder models are held')
    ARGS = parser.parse_args()
    models = get_all_models_in_dir(ARGS.dir)
