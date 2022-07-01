import os
from argparse import ArgumentParser
from gan_tester import run
import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd

def get_results(ARGS, gen=('edge_2d','tap'), dis=('edge_2d','tap'), data=('edge_2d', 'tap')):
    gen_model_dir = os.path.join(ARGS.dir, 'models/sim2real/alex/trained_gans/['+gen[0]+']/128x128_['+gen[1]+']_250epochs')
    discrim_model_dir = os.path.join(ARGS.dir, 'models/sim2real/alex/trained_gans/['+dis[0]+']/128x128_['+dis[1]+']_250epochs')
    real_images_dir = os.path.join(ARGS.dir, 'data/Bourne/tactip/real/'+data[0]+'/'+data[1]+'/csv_val/images')
    sim_images_dir = os.path.join(ARGS.dir, 'data/Bourne/tactip/sim/'+data[0]+'/'+data[1]+'/128x128/csv_val/images')
    return run(gen_model_dir, discrim_model_dir, real_images_dir, sim_images_dir, ARGS.dev)

def save_results(MSE, score, gen, discrim, data, csv_file='results/compare_existing_models.csv'):
    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))
    row_name = 'gen_'+str(gen)+'--dis_'+str(discrim)+'--data_'+str(data)
    if os.path.isfile(csv_file): # load existing data frame to add to
        df = pd.read_csv(csv_file, index_col=0)
        data = df.to_dict('index')
    else: # make new dataframe
        data = {}
    data[row_name] = {'MSE on Validation': MSE, 'Score on discriminator (accuracy)': score}  # add or overwrite new data
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
                MSEs, discrim_scores = get_results(ARGS, generator, discrim, data)
                save_results(np.mean(MSEs), np.mean(discrim_scores), generator, discrim, data)


            # print('Gen:', generator, ' Data: ', data)
            # print(np.mean(MSEs))
            # print(np.mean(discrim_scores))
            # print('=====================')
