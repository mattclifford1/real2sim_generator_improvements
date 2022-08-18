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
        Ys.append(df[cols[i]].values[1:])
    x = df['epoch'].values[1:]
    Ys = np.array(Ys)
    return x, np.mean(Ys, axis=0), np.std(Ys, axis=0)
    # return x, df[cols[i]].values[1:], np.std(Ys, axis=0)

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
        ## full data
        # str(int(data_limits[7]*100))+'% Data No GAN': os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[7])),
        # str(int(data_limits[7]*100))+'% Data GAN': os.path.join('surface_3d', 'shear', 'not_pretrained', 'GAN_'+train_routine+'_DS:'+str(data_limits[7])),

        str(int(data_limits[7]*100))+'% Data [transferred net] No GAN': os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[7])),
        str(int(data_limits[7]*100))+'% Data [transferred net] GAN': os.path.join('surface_3d', 'shear', pretrained, 'GAN_'+train_routine+'_DS:'+str(data_limits[7])),


        # 0.1 data
        # str(int(data_limits[2]*100))+'% Data No GAN': os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[2])),
        # str(int(data_limits[2]*100))+'% Data [transferred net] No GAN': os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[2])),
    }

    cols = ['mean training loss', 'val MSE', 'val_SSIM', 'Downstream MAE']
    cols = ['Downstream MAE']

    fig, ax = plt.subplots(nrows=1, ncols=len(cols), figsize=(8,6))
    i = 0
    if cols[i] == 'Downstream MAE':
        ax.plot(range(1,251), [0.041]*250, label='real pose net')
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
        ax.set_ylabel('MAE')
        ax.set_ylim(0.02, 0.3)
        ax.set_title('Downstream Task Performance')
    ax.legend()
    plt.show()
