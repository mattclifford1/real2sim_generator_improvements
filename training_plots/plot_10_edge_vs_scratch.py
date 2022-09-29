import matplotlib.pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np
import sys; sys.path.append('..'); sys.path.append('.')
from training_plots.plot_utils import get_avg_of_runs, plot_on_ax

'''run with eg:
$ python plotting/training_graphs_from_list.py --dir ~/Downloads/matt/

notes: surface_3d shear pose estimations MAES:
            - real   ->   0.041
            - sim    ->   0.026/ 0.021
'''

if __name__ == '__main__':
    parser = ArgumentParser(description='Plot training graphs')
    # parser.add_argument("--dir", default=os.path.join(os.path.expanduser('~'), 'summer-project', 'models', 'sim2real', 'matt'), help='path to folder where training graphs are within')
    # parser.add_argument("--dir", default=os.path.join('gan_models', 'training_stats-data_reduce'), help='path to folder where training graphs are within')
    parser.add_argument("--dir", default=os.path.join('gan_models', 'training_csvs'), help='path to folder where training graphs are within')
    parser.add_argument("--std", default=False, action='store_true', help='plot standard deviation of all runs')
    ARGS = parser.parse_args()

    # define the  label:filepath   to plot
    pretrained_1 = 'pretrained_edge_tap'
    pretrained_2 = 'pretrained_edge_shear'
    pretrained_3 = 'pretrained_surface_tap'
    train_routine = 'LR:0.0002_decay:0.1_BS:64'
    loss_type = 'no_gan'
    loss_type = 'GAN_'
    data_limits = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    cols = ['Downstream MAE']


    curves_to_plot = {
        ## ones to plot
        str(int(data_limits[7]*100))+'% Data': os.path.join('surface_3d', 'shear', 'not_pretrained', loss_type+train_routine+'_DS:'+str(data_limits[7])),
        str(int(data_limits[2]*100))+'% Data [Model Transfer]': os.path.join('surface_3d', 'shear', pretrained_2, loss_type+train_routine+'_DS:'+str(data_limits[2])),
        str(int(data_limits[2]*100))+'% Data': os.path.join('surface_3d', 'shear', 'not_pretrained', loss_type+train_routine+'_DS:'+str(data_limits[2]))
        }

    colours = [None, None, ["#9b59b6", "#3498db", "#e74c3c", "#2ecc71", "#95a5a6"], None]

    titles = ['Transfer from Edge Tap', 'Transfer from Edge Shear', 'No Transfer (Scratch)', 'Transfer from Surface Tap']



    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    plot_on_ax(ax, curves_to_plot, cols, ARGS.dir, ARGS.std, None)
    ax.set_title('Transfer from Edge vs Scratch')
    ax.set_title('')

    plt.tight_layout()
    plt.show()
