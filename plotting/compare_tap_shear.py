import matplotlib.pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np

'''run with eg:
$ python plotting/training_graphs_from_list.py --dir ~/Downloads/matt/

notes: surface_3d shear pose estimations MAES:
            - real   ->   0.041
            - sim    ->   0.026/ 0.021
'''

def get_avg_of_runs(dir, csv_file='training_stats.csv'):
    Ys = []
    for run in os.listdir(dir):
        path = os.path.join(dir, run, csv_file)
        df = pd.read_csv(path)
        if len(df) == 251:
            Ys.append(df[cols[i]].values[0:])
    x = df['epoch'].values[0:]
    Ys = np.array(Ys)
    return x, np.mean(Ys, axis=0), np.std(Ys, axis=0)
    # return x, df[cols[i]].values[1:], np.std(Ys, axis=0)

if __name__ == '__main__':
    parser = ArgumentParser(description='Plot training graphs')
    # parser.add_argument("--dir", default=os.path.join(os.path.expanduser('~'), 'summer-project', 'models', 'sim2real', 'matt'), help='path to folder where training graphs are within')
    parser.add_argument("--dir", default=os.path.join('gan_models', 'training_csvs'), help='path to folder where training graphs are within')
    parser.add_argument("--std", default=False, action='store_true', help='plot standard deviation of all runs')
    ARGS = parser.parse_args()

    # define the  label:filepath   to plot
    train_routine = 'LR:0.0002_decay:0.1_BS:64_DS:1.0'
    curves_to_plot = {
        'Scratch Surface Shear': os.path.join('surface_3d', 'shear', 'not_pretrained', 'GAN_'+train_routine),
        'Scratch Surface Tap': os.path.join('surface_3d', 'tap', 'not_pretrained', 'GAN_'+train_routine),
        'Transfer Surface Shear': os.path.join('surface_3d', 'shear', 'pretrained_surface_shear', 'GAN_'+train_routine),
        'Transfer Surface Tap': os.path.join('surface_3d', 'tap', 'pretrained_surface_shear', 'GAN_'+train_routine),


    }

    cols = ['mean training loss', 'val MSE', 'val_SSIM', 'Downstream MAE']
    cols = ['Downstream MAE']

    fig, ax = plt.subplots(nrows=1, ncols=len(cols), figsize=(8,6))
    i = 0
    if cols[i] == 'Downstream MAE':
        ax.plot(range(0,251), [0.041]*251, label='Expected Error on Real Data')
    for key in curves_to_plot.keys():
        dir = curves_to_plot[key]    # training run csv file of stats
        dir = os.path.join(ARGS.dir, dir)
        x, avg, std = get_avg_of_runs(dir)
        p = ax.plot(x, avg, label=key)
        if ARGS.std == True:
            colour = p[0].get_color()
            # ax.plot(x, avg+std, color=colour, alpha=0.7)
            # ax.plot(x, avg-std, color=colour, alpha=0.7)
            ax.fill_between(x, avg-std, avg+std, alpha=0.2, color=colour)

        ax.set_xlabel('Epoch')
    ax.set_title(cols[i])
    if cols[i] == 'Downstream MAE':
        ax.set_ylabel('Pose Estimation MAE')
        ax.set_ylim(0.02, 0.1)
        # ax.set_title('Downstream Task Performance')
    ax.legend()
    plt.show()
