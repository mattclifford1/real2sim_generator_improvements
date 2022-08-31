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

    curves_to_plot1 = {
    ## ones to plot
    str(int(data_limits[7]*100))+'% Data': os.path.join('surface_3d', 'shear', 'not_pretrained', loss_type+train_routine+'_DS:'+str(data_limits[7])),
    str(int(data_limits[2]*100))+'% Data [transferred edge tap]': os.path.join('surface_3d', 'shear', pretrained_1, loss_type+train_routine+'_DS:'+str(data_limits[2])),
    #
    }
    curves_to_plot2 = {
        ## ones to plot
        str(int(data_limits[7]*100))+'% Data': os.path.join('surface_3d', 'shear', 'not_pretrained', loss_type+train_routine+'_DS:'+str(data_limits[7])),
        str(int(data_limits[2]*100))+'% Data [transferred edge shear]': os.path.join('surface_3d', 'shear', pretrained_2, loss_type+train_routine+'_DS:'+str(data_limits[2])),
        #
    }
    curves_to_plot3 = {
        ## ones to plot
        str(int(data_limits[7]*100))+'% Data': os.path.join('surface_3d', 'shear', 'not_pretrained', loss_type+train_routine+'_DS:'+str(data_limits[7])),
        str(int(data_limits[2]*100))+'% Data': os.path.join('surface_3d', 'shear', 'not_pretrained', loss_type+train_routine+'_DS:'+str(data_limits[2])),
        #
    }
    curves_to_plot4 = {
        ## ones to plot
        str(int(data_limits[7]*100))+'% Data': os.path.join('surface_3d', 'shear', 'not_pretrained', loss_type+train_routine+'_DS:'+str(data_limits[7])),
        str(int(data_limits[2]*100))+'% Data [transferred surface tap]': os.path.join('surface_3d', 'shear', pretrained_3, loss_type+train_routine+'_DS:'+str(data_limits[7])),
        #
    }

    plts = [curves_to_plot1,
            curves_to_plot2,
            curves_to_plot3,
            curves_to_plot4]

    colours = [None, None, ["#9b59b6", "#3498db", "#e74c3c", "#2ecc71", "#95a5a6"], None]

    titles = ['Transfer from Edge Shear', 'Transfer from Edge Tap', 'No Transfer (Scratch)', 'Transfer from Surface Tap']

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
    c = 0
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            plot_on_ax(col, plts[c], cols, ARGS.dir, ARGS.std, colours[c])
            col.set_title(titles[c])
            c += 1
            if i == 0:
                col.set_xlabel('')
                col.set_xticks([])
            if j == 1:
                col.set_ylabel('')
                col.set_yticks([])

    plt.tight_layout()
    plt.show()
