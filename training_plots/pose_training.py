import matplotlib.pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser
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
    parser.add_argument("--dir", default=os.path.join(os.path.expanduser('~'), 'summer-project', 'models', 'pose_estimation'), help='path to folder where training graphs are within')
    ARGS = parser.parse_args()

    # define the  label:filepath   to plot
    train_routine = '_LR:0.0001_BS:16'
    curves_to_plot = {
        'Real': os.path.join('surface_3d', 'shear', 'real'+train_routine, 'run_0', 'training_stats.csv'),
        'Simulation': os.path.join('surface_3d', 'shear', 'sim'+train_routine, 'run_0', 'training_stats.csv'),
    }

    cols = ['mean training loss', 'val MAE']
    cols = ['val MAE']
    plot_runs(curves_to_plot, cols, ARGS.dir, ARGS.std)

    # if len(cols) == 1:  # can't iterate over one plot
    #     fig, ax = plt.subplots(nrows=1, ncols=len(cols), figsize=(8,6))
    #     i = 0
    #     for key in curves_to_plot.keys():
    #         file = curves_to_plot[key]    # training run csv file of stats
    #         df = pd.read_csv(os.path.join(ARGS.dir, file))
    #         ax.plot(df['epoch'].values[0:], df[cols[i]].values[0:], label=key)
    #         ax.set_xlabel('Epoch')
    #     ax.set_title(cols[i])
    #     # ax.set_xlim(1, 100)
    #     if cols[i] == 'Validation MAE':
    #         ax.set_ylabel('MAE')
    #         ax.set_ylim(0.02, 0.3)
    #         ax.set_title('Downstream Task Performance')
    #     ax.legend()
    # else:
    #     fig, ax = plt.subplots(nrows=1, ncols=len(cols), figsize=(17,11))
    #     for i, col in enumerate(ax):
    #         for key in curves_to_plot.keys():
    #             file = curves_to_plot[key]    # training run csv file of stats
    #             df = pd.read_csv(os.path.join(ARGS.dir, file))
    #             col.plot(df['epoch'].values[0:], df[cols[i]].values[0:], label=key)
    #             col.set_xlabel('Epoch')
    #             # ax.set_xlim(1, 100)
    #
    #         col.legend()
    #         col.set_title(cols[i])
    # plt.show()
