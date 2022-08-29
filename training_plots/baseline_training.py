import matplotlib.pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np
import sys; sys.path.append('..'); sys.path.append('.')
from training_plots.plot_utils import get_avg_of_runs, plot_runs


if __name__ == '__main__':
    parser = ArgumentParser(description='Plot training graphs')
    # parser.add_argument("--dir", default=os.path.join(os.path.expanduser('~'), 'summer-project', 'models', 'sim2real', 'matt'), help='path to folder where training graphs are within')
    parser.add_argument("--dir", default=os.path.join('gan_models', 'training_stats-data_reduce'), help='path to folder where training graphs are within')
    parser.add_argument("--std", default=True, action='store_false', help='plot standard deviation of all runs')
    ARGS = parser.parse_args()

    # define the  label:filepath   to plot
    train_routine = 'LR:0.0002_decay:0.1_BS:64'

    curves_to_plot = {
        'Surface 3D shear trained from scratch': os.path.join('surface_3d', 'shear', 'not_pretrained', 'GAN_'+train_routine+'_DS:1.0')
    }

    cols = ['mean training loss', 'val MSE', 'val_SSIM', 'Downstream MAE']
    cols = ['mean training loss', 'val_SSIM', 'Downstream MAE']
    # cols = ['Downstream MAE']
    plot_runs(curves_to_plot, cols, ARGS.dir, ARGS.std)
    
