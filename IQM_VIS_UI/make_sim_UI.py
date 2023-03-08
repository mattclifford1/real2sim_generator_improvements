import os
import numpy as np
import IQM_VIS
import sys; sys.path.append('..'); sys.path.append('.')
import data_holder
import image_utils


def run():
    file_path = os.path.dirname(os.path.abspath(__file__))

    # metrics functions must return a single value
    metric = {'MAE': IQM_VIS.metrics.MAE,
              'MSE': IQM_VIS.metrics.MSE,
              '1-SSIM': IQM_VIS.metrics.ssim()}

    # metrics images return a numpy image
    metric_images = {'MSE': IQM_VIS.metrics.MSE_image,
                     'SSIM': IQM_VIS.metrics.SSIM_image()}

    # first row of images
    sim_dataset_csv = os.path.join(os.path.expanduser('~'),'summer-project/data/Bourne/tactip/sim/surface_3d/shear/128x128/csv_train/targets.csv')

    data = data_holder.dataset_holder(sim_dataset_csv,
                                  metric,
                                  metric_images)
    # second row of images
    # define the transformations
    transformations = {
               'rotation':{'min':-180, 'max':180, 'function':IQM_VIS.transforms.rotation},    # normal input
               'blur':{'min':1, 'max':41, 'normalise':'odd', 'function':IQM_VIS.transforms.blur},  # only odd ints
               'brightness':{'min':-1.0, 'max':1.0, 'function':IQM_VIS.transforms.brightness},   # normal but with float
               'x_shift':{'min':-1.0, 'max':1.0, 'function':image_utils.translate_x},
               'y_shift':{'min':-1.0, 'max':1.0, 'function':image_utils.translate_y},
               'zoom':    {'min': 0.8, 'max':1.2, 'function':IQM_VIS.transforms.zoom_image, 'init_value': 1.0, 'num_values':21},  # requires non standard slider params
               }
    # define any parameters that the metrics need (names shared across both metrics and metric_images)
    ssim_params = {'sigma': {'min':0.25, 'max':5.25, 'init_value': 1.5},  # for the guassian kernel
                   # 'kernel_size': {'min':1, 'max':41, 'normalise':'odd', 'init_value': 11},  # ignored if guassian kernel used
                   'k1': {'min':0.01, 'max':0.21, 'init_value': 0.01},
                   'k2': {'min':0.01, 'max':0.21, 'init_value': 0.03}}

    # use the API to create the UI
    IQM_VIS.make_UI(data,
                transformations,
                metrics_avg_graph=True,
                metric_params=ssim_params)


if __name__ == '__main__':
    run()
