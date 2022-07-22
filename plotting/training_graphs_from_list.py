import matplotlib.pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser

'''run with eg:
$ python plotting/training_graphs_from_list.py --dir ~/Downloads/matt/
'''

if __name__ == '__main__':
    parser = ArgumentParser(description='Plot training graphs')
    parser.add_argument("--dir", default=os.path.join(os.path.expanduser('~'), 'Downloads', 'matt'), help='path to folder where training graphs are within')
    ARGS = parser.parse_args()

    # define the  label:filepath   to plot
    pretrained = 'pretrained_edge_tap'
    train_routine = 'LR:0.0002_decay:0.1_BS:64'
    data_limits = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    curves_to_plot = {
        # '''not pretrained - full data'''
        # 'edge tap': os.path.join('edge_2d', 'tap', 'not_pretrained', 'no_gan'+train_routine, 'run_0', 'training_stats.csv'),
        # 'edge tap GAN': os.path.join('edge_2d', 'tap', 'not_pretrained', 'GAN_'+train_routine, 'run_1', 'training_stats.csv'),
        # 'edge shear': os.path.join('edge_2d', 'shear', 'not_pretrained', 'no_gan'+train_routine, 'run_0', 'training_stats.csv'),
        # 'edge shear GAN': os.path.join('edge_2d', 'shear', 'not_pretrained', 'GAN_'+train_routine, 'run_1', 'training_stats.csv'),
        # 'surface tap': os.path.join('surface_3d', 'tap', 'not_pretrained', 'no_gan'+train_routine, 'run_0', 'training_stats.csv'),
        # 'surface tap GAN': os.path.join('surface_3d', 'tap', 'not_pretrained', 'GAN_'+train_routine, 'run_0', 'training_stats.csv'),
        # 'surface shear': os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine, 'run_0', 'training_stats.csv'),
        # 'surface shear GAN': os.path.join('surface_3d', 'shear', 'not_pretrained', 'GAN_'+train_routine, 'run_0', 'training_stats.csv'),

        # '''pretrained - full data'''
        # 'pretrained[et] edge tap': os.path.join('edge_2d', 'tap', pretrained, 'no_gan'+train_routine, 'run_0', 'training_stats.csv'),
        # 'pretrained[et] edge tap GAN': os.path.join('edge_2d', 'tap', pretrained, 'GAN_'+train_routine, 'run_0', 'training_stats.csv'),
        # 'pretrained[et] edge shear': os.path.join('edge_2d', 'shear', pretrained, 'no_gan'+train_routine, 'run_0', 'training_stats.csv'),
        # 'pretrained[et] edge shear GAN': os.path.join('edge_2d', 'shear', pretrained, 'GAN_'+train_routine, 'run_0', 'training_stats.csv'),
        # 'pretrained[et] surface tap': os.path.join('surface_3d', 'tap', pretrained, 'no_gan'+train_routine, 'run_0', 'training_stats.csv'),
        # 'pretrained[et] surface tap GAN': os.path.join('surface_3d', 'tap', pretrained, 'GAN_'+train_routine, 'run_0', 'training_stats.csv'),
        # 'pretrained[et] surface shear': os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine, 'run_0', 'training_stats.csv'),
        # 'pretrained[et] surface shear GAN': os.path.join('surface_3d', 'shear', pretrained, 'GAN_'+train_routine, 'run_0', 'training_stats.csv'),

        # '''not pretrained - limited data'''
        'surface shear '+str(data_limits[0]): os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[0]), 'run_0', 'training_stats.csv'),
        'surface shear '+str(data_limits[1]): os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[1]), 'run_0', 'training_stats.csv'),
        'surface shear '+str(data_limits[2]): os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[2]), 'run_0', 'training_stats.csv'),
        'surface shear '+str(data_limits[3]): os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[3]), 'run_0', 'training_stats.csv'),
        'surface shear '+str(data_limits[4]): os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[4]), 'run_0', 'training_stats.csv'),
        'surface shear '+str(data_limits[5]): os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[5]), 'run_0', 'training_stats.csv'),
        'surface shear '+str(data_limits[6]): os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[6]), 'run_0', 'training_stats.csv'),
        'surface shear '+str(data_limits[7]): os.path.join('surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[7]), 'run_0', 'training_stats.csv'),

        # '''pretrained - limited data'''
        # '[et]surface shear '+str(data_limits[0]): os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[0]), 'run_0', 'training_stats.csv'),
        # '[et]surface shear '+str(data_limits[1]): os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[1]), 'run_0', 'training_stats.csv'),
        # '[et]surface shear '+str(data_limits[2]): os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[2]), 'run_0', 'training_stats.csv'),
        # '[et]surface shear '+str(data_limits[3]): os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[3]), 'run_0', 'training_stats.csv'),
        # '[et]surface shear '+str(data_limits[4]): os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[4]), 'run_0', 'training_stats.csv'),
        # '[et]surface shear '+str(data_limits[5]): os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[5]), 'run_0', 'training_stats.csv'),
        # '[et]surface shear '+str(data_limits[6]): os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[6]), 'run_0', 'training_stats.csv'),
        # '[et]surface shear '+str(data_limits[7]): os.path.join('surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[7]), 'run_0', 'training_stats.csv'),



    }

    cols = ['mean training loss', 'val MSE', 'val_SSIM', 'Downstream MAE']
    fig, ax = plt.subplots(nrows=1, ncols=len(cols), figsize=(17,11))

    for i, col in enumerate(ax):
        for key in curves_to_plot.keys():
            file = curves_to_plot[key]
            df = pd.read_csv(os.path.join(ARGS.dir, file))
            # print(df['epoch'].values)
            col.plot(df['epoch'].values[1:], df[cols[i]].values[1:], label=key)
            # if i == 0:
            #     col.set_ylabel('epoch')
            col.set_xlabel('epoch')

        col.legend()
        col.set_title(cols[i])
    plt.show()
