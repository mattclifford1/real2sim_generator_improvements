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
    parser.add_argument("--dir", default=os.path.join('gan_models', 'training_csvs'), help='path to folder where training graphs are within')
    parser.add_argument("--std", default=False, action='store_true', help='plot standard deviation of all runs')
    ARGS = parser.parse_args()

    # define the  label:filepath   to plot
    data_type = 'shear'
    train_routine = 'LR:0.0002_decay:0.1_BS:64_DS:1.0'
    curves_to_plot = {
        'Scratch': os.path.join('edge_2d', data_type, 'not_pretrained', 'GAN_'+train_routine),
        'Edge Tap Transfer': os.path.join('edge_2d', data_type, 'pretrained_edge_tap', 'GAN_'+train_routine),
        # 'Edge Shear Transfer': os.path.join('edge_2d', data_type, 'pretrained_edge_shear', 'GAN_'+train_routine),
        'Surface Tap Transfer': os.path.join('edge_2d', data_type, 'pretrained_surface_tap', 'GAN_'+train_routine),
        # 'Surface Shear Transfer': os.path.join('edge_2d', data_type, 'pretrained_surface_shear', 'GAN_'+train_routine),
    }

    cols = ['mean training loss', 'val MSE', 'val_SSIM', 'Downstream MAE']
    cols = ['Downstream MAE']

    plot_runs(curves_to_plot, cols, ARGS.dir, ARGS.std)
