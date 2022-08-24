import matplotlib.pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np
import sys; sys.path.append('..'); sys.path.append('.')
from training_plots.plot_utils import get_avg_of_runs

'''run with eg:
$ python plotting/training_graphs_from_list.py --dir ~/Downloads/matt/

notes: surface_3d shear pose estimations MAES:
            - real   ->   0.0412
            - sim    ->   0.023
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
        'Scratch': os.path.join('surface_3d', data_type, 'not_pretrained', 'GAN_'+train_routine),
        # 'Surface Tap Transfer': os.path.join('surface_3d', data_type, 'pretrained_surface_tap', 'GAN_'+train_routine),
        'Surface Shear Transfer': os.path.join('surface_3d', data_type, 'pretrained_surface_shear', 'GAN_'+train_routine),
        # 'Edge Tap Transfer': os.path.join('surface_3d', data_type, 'pretrained_edge_tap', 'GAN_'+train_routine),
        'Edge Shear Transfer': os.path.join('surface_3d', data_type, 'pretrained_edge_shear', 'GAN_'+train_routine),
    }

    cols = ['mean training loss', 'val MSE', 'val_SSIM', 'Downstream MAE']
    cols = ['Downstream MAE']

    fig, ax = plt.subplots(nrows=1, ncols=len(cols), figsize=(8,6))
    i = 0
    colours = ["#9b59b6", "#3498db", "#e74c3c", "#2ecc71", "#34495e", "#95a5a6"]
    colour_num = 0
    if cols[i] == 'Downstream MAE':
        ax.plot(range(0,251), [0.0412]*251, label='Expected Error on Real Data', color=colours[colour_num])
        colour_num += 1
        ax.plot(range(0,251), [0.023]*251, label='Expected Error on Simulated Data', color=colours[colour_num])
        colour_num += 1
    for key in curves_to_plot.keys():
        dir = curves_to_plot[key]    # training run csv file of stats
        dir = os.path.join(ARGS.dir, dir)
        x, avg, std = get_avg_of_runs(dir, cols[i])
        p = ax.plot(x, avg, label=key, color=colours[colour_num])
        colour_num += 1
        if ARGS.std == True:
            colour = p[0].get_color()
            # ax.plot(x, avg+std, color=colour, alpha=0.7)
            # ax.plot(x, avg-std, color=colour, alpha=0.7)
            ax.fill_between(x, avg-std, avg+std, alpha=0.2, color=colour)

        ax.set_xlabel('Epoch')
    # ax.set_title(cols[i])
    if cols[i] == 'Downstream MAE':
        ax.set_ylabel('Pose Estimation MAE')
        ax.set_ylim(0.02, 0.1)
        # ax.set_title('Downstream Task Performance')
    ax.legend()
    plt.show()
