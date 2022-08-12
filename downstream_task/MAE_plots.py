'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

plot MAE over the validation set for t PoseNet
'''
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import sys; sys.path.append('..'); sys.path.append('.')
import downstream_task.networks.model_128 as m_128
from downstream_task.data import dataloader
from downstream_task.pose_net_utils import load_weights
from image_transformations import networks, image_utils  # change path for this eventually
from image_transformations.image_utils import change_im, load_image  # change path for this eventually

class make_plots():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sensor_data = {}
        self.generator = networks.generator(self.args.generator_path)
        self.pose_esimator_sim = networks.pose_estimation(self.args.pose_path, sim=True)
        self.pose_esimator_real = networks.pose_estimation(self.args.pose_path, sim=False)
        self.load_dataset(self.args.csv_path)

    def load_dataset(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.im_num = 0   # row of csv dataset to use
        self.im_sim_dir = os.path.join(os.path.dirname(csv_file), 'images')
        self.im_real_dir = os.path.join(os.path.dirname(image_utils.get_real_csv_given_sim(csv_file)), 'images')

    def get_preds(self, i):
        self.im_num = i
        self.load_sim_image()
        self.load_real_image()
        preds = {}
        preds['P(Xs)'] = self.pose_esimator_sim.get_prediction(self.sensor_data['Xs']).cpu().detach().numpy()
        preds['P(G(Xr))'] = self.pose_esimator_sim.get_prediction(self.sensor_data['G(Xr)']).cpu().detach().numpy()
        preds['P(Xr)'] = self.pose_esimator_real.get_prediction(self.sensor_data['Xr']).cpu().detach().numpy()
        labels = self.pose_esimator_sim.normalise_y_labels(self.sensor_data['Y']).cpu().detach().numpy()
        return preds, labels

    def run(self):
        poses = self.pose_esimator_sim.y_names
        fig, ax = plt.subplots(nrows=1, ncols=len(poses), figsize=(12,5))
        data_size = 5 if self.args.dev else len(self.df)
        for i in tqdm(range(data_size)):
            preds, labels = self.get_preds(i)
            for j, col in enumerate(ax):
                col.scatter(labels[j], preds['P(Xr)'][0][j], color='g', s=0.5)
                col.scatter(labels[j], preds['P(Xs)'][0][j], color='b', s=0.5)
                col.scatter(labels[j], preds['P(G(Xr))'][0][j], color='r', s=0.5)
        for j, col in enumerate(ax):
            col.set_title(poses[j])
            x = [-1, 1]
            col.plot(x, x, color='black')
        plt.show()



    '''
    image loaders
    '''
    def load_sim_image(self):
        image = self.df.iloc[self.im_num]['sensor_image']
        image_path = os.path.join(self.im_sim_dir, image)
        poses = ['pose_'+str(i) for i in range(1,7)]
        self.sensor_data['Y'] = {}
        for pose in poses:
            self.sensor_data['Y'][pose] = self.df.iloc[self.im_num][pose]
        self.sensor_data['Xs'] = load_image(image_path)
        self.sensor_data['Xs'] = image_utils.process_im(self.sensor_data['Xs'], data_type='sim')

    def load_real_image(self):
        image_path = os.path.join(self.im_real_dir, self.df.iloc[self.im_num]['sensor_image'])
        self.sensor_data['im_raw'] = image_utils.load_and_crop_raw_real(image_path)

        self.sensor_data['Xr'] = image_utils.process_im(self.sensor_data['im_raw'], data_type='real')
        self.sensor_data['G(Xr)'] = self.generator.get_prediction(self.sensor_data['Xr'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='image file to use', default=os.path.join(os.path.expanduser('~'),'summer-project/data/Bourne/tactip/sim/surface_3d/tap/128x128/csv_train/images/image_1.png'))
    parser.add_argument('--csv_path', type=str, help='csv file to use', default=os.path.join(os.path.expanduser('~'),'summer-project/data/Bourne/tactip/sim/surface_3d/shear/128x128/csv_train/targets.csv'))
    parser.add_argument('--generator_path', type=str, help='generator weights file to use', default=os.path.join(os.path.expanduser('~'),'summer-project/models/sim2real/matt/surface_3d/shear/pretrained_edge_tap/no_ganLR:0.0002_decay:0.1_BS:64_DS:1.0/run_0/models/best_generator.pth'))
    parser.add_argument('--pose_path', type=str, help='pose net weights file to use', default=os.path.join(os.path.expanduser('~'), 'summer-project/models/pose_estimation/surface_3d/shear/sim_LR:0.0001_BS:16/run_0/checkpoints/best_model.pth'))
    parser.add_argument("--dev", default=False, action='store_true', help='dev runs on small datasize')

    args = parser.parse_args()

    plotter = make_plots(args)
    plotter.run()
