import sys
import argparse
from functools import partial
import os
import pandas as pd
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys; sys.path.append('..'); sys.path.append('.')
from image_utils import change_im, load_image
import image_utils
import networks
import metrics


class loop_transforms():
    def __init__(self, args):
        self.args = args
        self.generator = networks.generator(self.args.generator_path)
        self.pose_esimator_sim = networks.pose_estimation(self.args.pose_path, sim=True)
        self.pose_esimator_real = networks.pose_estimation(self.args.pose_path, sim=False)
        self.metrics = metrics.im_metrics()
        self.sensor_data = {}
        self.load_dataset(self.args.csv_path)
        self.run()

    def load_dataset(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.im_nums = 5 if self.args.dev else len(self.df)
        self.im_num = 0   # row of csv dataset to use
        self.im_sim_dir = os.path.join(os.path.dirname(csv_file), 'images')
        self.im_real_dir = os.path.join(os.path.dirname(image_utils.get_real_csv_given_sim(csv_file)), 'images')
        self.load_sim_image()
        # self.load_real_image()

    def load_sim_image(self):
        image = self.df.iloc[self.im_num]['sensor_image']
        image_path = os.path.join(self.im_sim_dir, image)
        poses = ['pose_'+str(i) for i in range(1,7)]
        self.sensor_data['poses'] = {}
        for pose in poses:
            self.sensor_data['poses'][pose] = self.df.iloc[self.im_num][pose]
        self.sensor_data['Xs'] = load_image(image_path)
        self.sensor_data['Xs'] = image_utils.process_im(self.sensor_data['Xs'], data_type='sim')

    def load_real_image(self):
        image_path = os.path.join(self.im_real_dir, self.df.iloc[self.im_num]['sensor_image'])
        self.sensor_data['im_raw'] = image_utils.load_and_crop_raw_real(image_path)

        self.sensor_data['Xr'] = image_utils.process_im(self.sensor_data['im_raw'], data_type='real')
        self.sensor_data['G(Xr)'] = self.generator.get_prediction(self.sensor_data['Xr'])

    def run(self):
        metrics = self.metrics.get_metrics(self.sensor_data['Xs'], self.sensor_data['Xs'], ssim_image=False)
        metrics_keys = metrics.keys()
        results = {}
        for trans_param in tqdm(np.linspace(self.args.min, self.args.max, num=self.args.steps)):
            results[trans_param] = {}
            results[trans_param]['PoseNet'] = []
            for key in metrics_keys:
                results[trans_param][key] = []

            for i in tqdm(range(self.im_nums), leave=False):
                self.im_num = i
                self.load_sim_image()
                # self.load_real_image()
                sim_trans = image_utils.transform_image(self.sensor_data['Xs'], {self.args.trans: trans_param})

                metrics = self.metrics.get_metrics(self.sensor_data['Xs'], sim_trans, ssim_image=False)
                for key in metrics_keys:
                    results[trans_param][key].append(metrics[key][0])

                pose_error = self.pose_esimator_sim.get_error(sim_trans, self.sensor_data['poses'])
                results[trans_param]['PoseNet'].append(pose_error['MAE'][0])

        data = {self.args.trans: [], 'PoseNet':[]}
        for key in metrics_keys:
            data[key] = []
        for key in results.keys():
            data[self.args.trans].append(key)
            for k in results[key].keys():
                data[k].append(np.array(results[key][k]).mean())

        df = pd.DataFrame.from_dict(data)
        df.to_csv(os.path.join('image_transformations', 'results', self.args.trans+'_'+str(self.args.min)+'_'+str(self.args.max)+'_'+str(self.args.steps)+'.csv'))
        df.plot.line(x=self.args.trans, y=['PoseNet', 'SSIM', 'MSSIM', 'NLPD', 'MSE', 'MAE', 'LPIPS_vgg'])
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='csv file to use', default=os.path.join(os.path.expanduser('~'),'summer-project/data/Bourne/tactip/sim/surface_3d/shear/128x128/csv_train/targets.csv'))
    parser.add_argument('--generator_path', type=str, help='generator weights file to use', default=os.path.join(os.path.expanduser('~'),'summer-project/models/sim2real/matt/surface_3d/shear/pretrained_edge_tap/no_ganLR:0.0002_decay:0.1_BS:64_DS:1.0/run_0/models/best_generator.pth'))
    parser.add_argument('--pose_path', type=str, help='pose net weights file to use', default=os.path.join(os.path.expanduser('~'), 'summer-project/models/pose_estimation/surface_3d/shear/sim_LR:0.0001_BS:16/run_0/checkpoints/best_model.pth'))
    parser.add_argument("--dev", default=False, action='store_true', help='run on limited data')
    parser.add_argument('--trans', type=str, help='image transform', default='rotation')
    parser.add_argument('--min', type=float, help='image transform min value', default=-40)
    parser.add_argument('--max', type=float, help='image transform max value', default=40)
    parser.add_argument('--steps', type=int, help='number of values to test (linspace)', default=10)

    args = parser.parse_args()

    loop_transforms(args)
