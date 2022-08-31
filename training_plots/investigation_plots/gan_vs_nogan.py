import matplotlib.pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np
import sys; sys.path.append('..'); sys.path.append('.')
from training_plots.plot_utils import get_avg_of_runs, plot_runs

'''run with eg:
$ python plotting/training_graphs_from_list.py --dir ~/Downloads/matt/

notes: surface_3d shear pose estimations MAES:
            - real   ->   0.041
            - sim    ->   0.026/ 0.021
'''

if __name__ == '__main__':
    parser = ArgumentParser(description='Plot training graphs')
    # parser.add_argument("--dir", default=os.path.join(os.path.expanduser('~'), 'summer-project', 'models', 'sim2real', 'matt'), help='path to folder where training graphs are within')
    parser.add_argument("--dir", default=os.path.join('gan_models', 'training_stats-data_reduce'), help='path to folder where training graphs are within')
    parser.add_argument("--std", default=False, action='store_true', help='plot standard deviation of all runs')
    ARGS = parser.parse_args()

    # define the  label:filepath   to plot
    pretrained = 'pretrained_edge_tap'
    train_routine = 'LR:0.0002_decay:0.1_BS:64'
    data_limits = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    curves_to_plot = {

# ======full data  ===============================================
        str(int(data_limits[7]*100))+'% Data No GAN': os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[7])),
        str(int(data_limits[7]*100))+'% Data GAN': os.path.join('surface_3d', 'shear', 'not_pretrained', 'GAN_'+train_routine+'_DS:'+str(data_limits[7])),

        # str(int(data_limits[7]*100))+'% Data [transferred net] No GAN': os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[7])),
        # str(int(data_limits[7]*100))+'% Data [transferred net] GAN': os.path.join('surface_3d', 'shear', pretrained, 'GAN_'+train_routine+'_DS:'+str(data_limits[7])),

# ===== 0.25 data  ===============================================
        # str(int(data_limits[3]*100))+'% Data No GAN': os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[3])),
        # str(int(data_limits[3]*100))+'% Data GAN': os.path.join('surface_3d', 'shear', 'not_pretrained', 'GAN_'+train_routine+'_DS:'+str(data_limits[3])),

        # str(int(data_limits[3]*100))+'% Data [transferred net] No GAN': os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[3])),
        # str(int(data_limits[3]*100))+'% Data [transferred net] GAN': os.path.join('surface_3d', 'shear', pretrained, 'GAN_'+train_routine+'_DS:'+str(data_limits[3])),

# ===== 0.1 data ===============================================
        # str(int(data_limits[2]*100))+'% Data No GAN': os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[2])),
        # str(int(data_limits[2]*100))+'% Data GAN': os.path.join('surface_3d', 'shear', 'not_pretrained', 'GAN_'+train_routine+'_DS:'+str(data_limits[2])),

        # str(int(data_limits[2]*100))+'% Data [transferred net] No GAN': os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[2])),
        # str(int(data_limits[2]*100))+'% Data [transferred net] GAN': os.path.join('surface_3d', 'shear', pretrained, 'GAN_'+train_routine+'_DS:'+str(data_limits[2])),

    }

    cols = ['mean training loss', 'val MSE', 'val_SSIM', 'Downstream MAE']
    cols = ['Downstream MAE']

    plot_runs(curves_to_plot, cols, ARGS.dir, ARGS.std)
