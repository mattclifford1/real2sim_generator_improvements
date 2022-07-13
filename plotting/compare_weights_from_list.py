import os
from argparse import ArgumentParser
import sys; sys.path.append('..'); sys.path.append('.')
from gan_tester import get_weights
import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def weights_diff(weights1, weights2):
    if weights1.keys() != weights2.keys():
        raise Exception('Weights need to have the same layer names to compare')
    layer_diffs = {}
    total_diff = 0
    for layer_name in weights1.keys():
        layer1 = weights1[layer_name]
        layer2 = weights2[layer_name]
        mean_layer_diff = np.abs(layer1 - layer2).mean()
        total_diff += mean_layer_diff
        layer_diffs[layer_name] = mean_layer_diff
    return layer_diffs, total_diff

def plot_avg_nets(networks, nets_weights, groupby):
    # compare weights
    plot_data = []
    for gen1 in generators:
        group_data = [gen1[0][:-3]+'_'+gen1[1]]
        exp_order = []
        for gen2 in generators:
            diff, total_diff = weights_diff(nets_weights[gen1], nets_weights[gen2])
            group_data.append(total_diff)
            exp_order.append(gen2[0][:-3]+'_'+gen2[1])
        plot_data.append(group_data)
    plot_df = pd.DataFrame(plot_data, columns=[groupby]+exp_order)
    # plot grouped bar chart
    plot_df.plot(x=groupby,
            kind='bar',
            stacked=False,
            # title=ARGS.metric+' with Differing '+inner_groupby
            )
    plt.ylabel('Weights mean abs difference')
    plt.xticks(rotation=0)
    plt.legend(loc='right', title='Network')
    save_file = 'results/weights_diff_'+groupby+'.png'
    plt.savefig(save_file)
    print('Saved figure to: '+save_file)

def plot_layers_diff(net_names, nets_weights, comparision_name, ax=None, y=False):
    plot_data = []
    nets = []
    for net_name in net_names:
        nets.append(net_name)
        group_data = []
        diff, total_diff = weights_diff(nets_weights[comparision_name], nets_weights[net_name])
        exp_order = []
        for layer in diff.keys():
            layer_name = layer.split('.')[0]
            if comparision_name == net_name:
                group_data.append(layer_name)
            else:
                group_data.append(diff[layer])
            exp_order.append(layer_name)
        plot_data.append(group_data)
    plot_df = pd.DataFrame(plot_data, columns=exp_order)
    plot_df = plot_df.T
    plot_df.columns = nets
    # plot grouped bar chart
    plot_df.plot(x=comparision_name,
            kind='bar',
            stacked=False,
            title=comparision_name,
            ax=ax
            )
    if ax is None:
        ax = plt
    # ax.tick_params('x', labelrotation=45)
    if y:
        ax.set_ylabel('Layer mean abs difference')
    ax.set_xlabel('Layer Name')
    ax.legend(loc='upper right', title='Network')



if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='Plot training graphs')
    parser.add_argument("--dir", default='..', help='path to folder where training graphs are within')
    ARGS = parser.parse_args()

    # define the  label:filepath   to plot
    models = {
        'edge tap': os.path.join('edge_2d', 'tap', 'not_pretrained', 'no_ganLR:0.001_decay:0.1_BS:64', 'run_0', 'models', 'best_generator.pth'),
        # 'edge tap GAN': os.path.join('edge_2d', 'tap', 'not_pretrained', 'GAN_LR:0.001_decay:0.1_BS:64', 'run_0', 'models', 'best_generator.pth'),
        'edge shear': os.path.join('edge_2d', 'shear', 'not_pretrained', 'no_ganLR:0.001_decay:0.1_BS:64', 'run_0', 'models', 'best_generator.pth'),
        # 'edge shear GAN': os.path.join('edge_2d', 'shear', 'not_pretrained', 'GAN_LR:0.001_decay:0.1_BS:64', 'run_0', 'models', 'best_generator.pth'),
        'surface tap': os.path.join('surface_3d', 'tap', 'not_pretrained', 'no_ganLR:0.001_decay:0.1_BS:64', 'run_0', 'models', 'best_generator.pth'),
        # 'surface tap GAN': os.path.join('surface_3d', 'tap', 'not_pretrained', 'GAN_LR:0.001_decay:0.1_BS:64', 'run_0', 'models', 'best_generator.pth'),
        'surface shear': os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_ganLR:0.001_decay:0.1_BS:64', 'run_0', 'models', 'best_generator.pth'),
        # 'surface shear GAN': os.path.join('surface_3d', 'shear', 'not_pretrained', 'GAN_LR:0.001_decay:0.1_BS:64', 'run_0', 'models', 'best_generator.pth'),
    }

    # get weights for all networks
    nets_weights = {}
    for key in tqdm(models.keys(), desc="Generators", leave=False):
        model_filepath = os.path.join(ARGS.dir, models[key])
        layers = get_weights(model_filepath)
        nets_weights[key] = layers

    fig, ax = plt.subplots(nrows=1, ncols=len(models.keys()))
    gen = 0
    y = True
    for col in ax:
        plot_layers_diff(list(models.keys()), nets_weights, list(models.keys())[gen], ax=col, y=y)
        gen += 1
        y = False
    # fig.tight_layout()
    fig.set_size_inches(4.5*len(models.keys()),7)
    save_file = 'results/weights_diff_layers.png'
    # plt.savefig(save_file, dpi=600)
    plt.show()
    print('Saved figure to: '+save_file)

    # plot_avg_nets(generators, nets_weights, groupby='Generator')
