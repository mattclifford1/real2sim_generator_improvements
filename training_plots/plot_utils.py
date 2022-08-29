import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_avg_of_runs(dir, col_to_get='Downstream MAE', csv_file='training_stats.csv'):
    Ys = []
    for run in os.listdir(dir):
        path = os.path.join(dir, run, csv_file)
        df = pd.read_csv(path)
        if len(df) == 251:
            Ys.append(df[col_to_get].values[0:])
            x = df['epoch'].values[0:]
    Ys = np.array(Ys)

    print('lowest all : ', col_to_get, np.min(Ys, axis=1).min())
    print('highest all: ', col_to_get, np.min(Ys, axis=1).max())
    print('lowest mean: ', col_to_get, np.mean(Ys, axis=0).min())
    print('lowest std : ', col_to_get, np.min(Ys, axis=1).std())

    return x, np.mean(Ys, axis=0), np.std(Ys, axis=0)


def plot_single(i, ax, curves_to_plot, cols, base_dir, plot_std=False):
    colours = ["#9b59b6", "#3498db", "#e74c3c", "#34495e", "#2ecc71", "#95a5a6"]
    colour_num = 0
    if cols[i] == 'Downstream MAE':
        ax.plot(range(0,251), [0.0412]*251, label='Expected Error on Real Data', color=colours[colour_num])
        colour_num += 1
        ax.plot(range(0,251), [0.023]*251, label='Expected Error on Simulated Data', color=colours[colour_num])
        colour_num += 1
    for key in curves_to_plot.keys():
        dir = curves_to_plot[key]    # training run csv file of stats
        dir = os.path.join(base_dir, dir)
        x, avg, std = get_avg_of_runs(dir, cols[i])
        if 'train' in cols[i]:
            p = ax.plot(x[1:], avg[1:], label=key, color=colours[colour_num])
        else:
            p = ax.plot(x, avg, label=key, color=colours[colour_num])
        colour_num += 1
        if plot_std == True:
            colour = p[0].get_color()
            # ax.plot(x, avg+std, color=colour, alpha=0.7)
            # ax.plot(x, avg-std, color=colour, alpha=0.7)
            if 'train' in cols[i]:
                ax.fill_between(x[1:], (avg-std)[1:], (avg+std)[1:], alpha=0.2, color=colour)
            else:
                ax.fill_between(x, avg-std, avg+std, alpha=0.2, color=colour)

        ax.set_xlabel('Epoch')
    ax.set_title(cols[i])
    if cols[i] == 'Downstream MAE':
        ax.set_ylabel('Pose Estimation MAE')
        ax.set_ylim(0.02, 0.3)
        ax.set_ylim(0.02, 0.2)
        # ax.set_ylim(0.02, 0.1)
        ax.set_title('PoseNet validation MAE ')
    elif cols[i] == 'val_SSIM':
        ax.set_title('Validation SSIM')
        ax.set_ylabel('SSIM')
    elif cols[i] == 'mean training loss':
        ax.set_title('Training Loss')
        ax.set_ylabel('Loss')
    ax.legend()


def plot_runs(curves_to_plot, cols, base_dir, plot_std=False):
    if len(cols) == 1:  # can't iterate over one plot
        fig, ax = plt.subplots(nrows=1, ncols=len(cols), figsize=(8,6))
        i = 0
        plot_single(i, ax, curves_to_plot, cols, base_dir, plot_std)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=len(cols), figsize=(13,4))
        for i, col in enumerate(ax):
            plot_single(i, col, curves_to_plot, cols, base_dir, plot_std)
    plt.tight_layout()
    plt.show()
