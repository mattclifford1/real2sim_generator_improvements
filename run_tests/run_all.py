import os
from argparse import ArgumentParser
import sys
sys.path.append('..')
sys.path.append('.')
from gan_tester import run
import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd

def get_results(ARGS, gen=('edge_2d','tap'), data=('edge_2d', 'tap')):
    gen_model_dir = os.path.join(ARGS.dir, 'models/sim2real/alex/trained_gans/['+gen[0]+']/128x128_['+gen[1]+']_250epochs')
    real_images_dir = os.path.join(ARGS.dir, 'data/Bourne/tactip/real/'+data[0]+'/'+data[1]+'/csv_val/images')
    sim_images_dir = os.path.join(ARGS.dir, 'data/Bourne/tactip/sim/'+data[0]+'/'+data[1]+'/128x128/csv_val/images')
    return run(gen_model_dir, real_images_dir, sim_images_dir, ARGS.dev)

def save_results(metrics_dict, gen, discrim, data, csv_file='results/compare_existing_models.csv'):
    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))
    # row_name = 'GEN_'+gen[0]+'_'+gen[1]+'--DIS_'+discrim[0]+'_'+discrim[1]+'--DATA_'+data[0]+'_'+data[1]
    row_name = {'Generator':gen[0][:-3]+'_'+gen[1],    #:-3 removes the final _2d or _3d for simplification
                'Discriminator':discrim[0][:-3]+'_'+discrim[1],
                'Data':data[0][:-3]+'_'+data[1]}
    if os.path.isfile(csv_file): # load existing data frame to add to
        df = pd.read_csv(csv_file, index_col=0)
        data = df.to_dict('index')
    else: # make new dataframe
        data = {}
    data[str(row_name)] = metrics_dict  # add or overwrite new data
    df = pd.DataFrame.from_dict(data, orient='index')
    df.to_csv(csv_file, index=True)



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
        for discrim in tqdm(discrims, desc="Discriminators", leave=False):
            for data in tqdm(datas, desc="Data", leave=False):
                metrics_dict = get_results(ARGS, generator, discrim, data)
                # take average all metrics
                avg_metrics = {}
                for key in metrics_dict.keys():
                    avg_metrics[key] = np.mean(metrics_dict[key])
                save_results(avg_metrics, generator, discrim, data)


            # print('Gen:', generator, ' Data: ', data)
            # print(np.mean(MSEs))
            # print(np.mean(discrim_scores))
            # print('=====================')
