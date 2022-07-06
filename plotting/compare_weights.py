import os
from argparse import ArgumentParser
import sys; sys.path.append('..'); sys.path.append('.')
from gan_tester import get_weights
import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def get_results(ARGS, gen=('edge_2d','tap'), dis=('edge_2d','tap')):
    gen_model_dir = os.path.join(ARGS.dir, 'models/sim2real/alex/trained_gans/['+gen[0]+']/128x128_['+gen[1]+']_250epochs')
    discrim_model_dir = os.path.join(ARGS.dir, 'models/sim2real/alex/trained_gans/['+dis[0]+']/128x128_['+dis[1]+']_250epochs')
    return get_weights(gen_model_dir, discrim_model_dir)

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

def plot_layers_diff(networks, nets_weights, comp_net):
    plot_data = []
    comparision_name = comp_net[0][:-3]+'_'+comp_net[1]
    nets = []
    for gen in generators:
        net_name = gen[0][:-3]+'_'+gen[1]
        nets.append(net_name)
        group_data = []
        diff, total_diff = weights_diff(nets_weights[comp_net], nets_weights[gen])
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
    print(plot_df)
    # plot grouped bar chart
    plot_df.plot(x=comparision_name,
            kind='bar',
            stacked=False,
            title=comparision_name
            )
    plt.ylabel('Layer mean abs difference')
    plt.xlabel('Layer Name')
    plt.legend(loc='right', title='Network')
    save_file = 'results/weights_diff_layers_'+comparision_name+'.png'
    plt.savefig(save_file)
    print('Saved figure to: '+save_file)


if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    parser.add_argument("--dev",  default=False, action='store_true', help='only run for a few samples when in dev mode')
    ARGS = parser.parse_args()

    # get all combinations of models
    task = ['edge_2d','surface_3d']
    sampling = ['tap', 'shear']
    generators = list(itertools.product(task, sampling))

    # get weights for all networks
    nets_weights = {}
    for generator in tqdm(generators, desc="Generators", leave=False):
        layers = get_results(ARGS, generator)
        nets_weights[generator] = layers
    # plot_avg_nets(generators, nets_weights, groupby='Generator')
    plot_layers_diff(generators, nets_weights, generators[0])
