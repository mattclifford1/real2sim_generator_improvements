''' image similarity metric runner'''
import sys; sys.path.append('..'); sys.path.append('.')
import torch
import torch.nn as nn
import numpy as np
import warnings

from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
# from torchmetrics import UniversalImageQualityIndex as UIQI
from torchmetrics.functional import universal_image_quality_index as UIQI
from torchmetrics import SpectralDistortionIndex as SDI
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from expert.pyramids import LaplacianPyramid

def preprocess_numpy_image(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)   # make into batch one 1
    image = torch.from_numpy(image)
    return image

class grey_to_3_channel_input():
    def __init__(self, func):
        self.func = func

    def __call__(self, x1, x2):
        # make inputs 3 channel if greyscale
        if x1.shape[1] == 1:
            x1 = torch.cat([x1, x1, x1], dim=1)
        if x2.shape[1] == 1:
            x2 = torch.cat([x2, x2, x2], dim=1)
        return self.func(x1, x2)

class im_metrics():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._get_similarity_metrics()

    def _get_similarity_metrics(self):
        # dictionary of callable metric functions
        # eg: score = self.metrics['xx'](im1, im2)
        self.metrics = {}
        self.metrics['MAE'] = nn.L1Loss()
        self.metrics['MSE'] = nn.MSELoss()
        self.metrics['NLPD'] = LaplacianPyramid(k=1).to(self.device)
        # self.metrics['NLPD_2'] = LaplacianPyramid(k=2).to(self.device)
        # self.metrics['NLPD_3'] = LaplacianPyramid(k=3).to(self.device)
        self.metrics['PSNR'] = PSNR().to(self.device)
        self.metrics['UIQI'] = UIQI
        # self.metrics['SDI'] = SDI() # gives nan
        # self.metrics['LPIPS_alex'] = grey_to_3_channel_input(LPIPS(net_type='alex').to(self.device))
        with warnings.catch_warnings():    # we don't care about the warnings these give
            warnings.simplefilter("ignore")
            self.metrics['LPIPS_vgg'] = grey_to_3_channel_input(LPIPS(net_type='vgg').to(self.device))
            self.metrics['SSIM'] = SSIM()
            self.metrics['MSSIM'] = MSSIM(kernel_size=7)

    def get_metrics(self, im_ref, im_comp):
        im_ref = preprocess_numpy_image(im_ref).to(device=self.device, dtype=torch.float)
        im_comp = preprocess_numpy_image(im_comp).to(device=self.device, dtype=torch.float)
        _scores = {}
        for key in self.metrics.keys():
            _scores[key] = []
        for key in self.metrics.keys():
            if key == 'MSSIM':
                _score = self.metrics[key](torch.nn.functional.interpolate(im_ref, 256),
                                           torch.nn.functional.interpolate(im_comp, 256))
            else:
                _score = self.metrics[key](im_ref, im_comp)
            _scores[key].append(_score.cpu().detach().numpy())
            if key == 'MSSIM' or key == 'SSIM':
                self.metrics[key].reset()   # clear mem buffer to stop overflow
        return _scores

if __name__ == '__main__':
    import os
    import pandas as pd
    import sys; sys.path.append('..'); sys.path.append('.')
    import gui_utils
    sensor_data = {}
    csv_file = os.path.join(os.path.expanduser('~'),'summer-project/data/Bourne/tactip/sim/surface_3d/shear/128x128/csv_train/targets.csv')
    df = pd.read_csv(csv_file)
    im_sim_dir = os.path.join(os.path.dirname(csv_file), 'images')
    im_num = 10
    image = df.iloc[im_num]['sensor_image']
    image_path = os.path.join(im_sim_dir, image)

    sensor_data['im_reference'] = gui_utils.load_image(image_path)
    sensor_data['im_reference'] = gui_utils.process_im(sensor_data['im_reference'], data_type='sim')
    im_comp = np.clip(sensor_data['im_reference']+0.3, 0, 1)


    # m = im_metrics()
    # metrics = m.get_metrics(sensor_data['im_reference'], im_comp)
    # print(metrics)

    mssim = MSSIM()#kernel_size=3, sigma=1.0,betas=(0.01, 0.1, 0.3, 0.2, 0.1),)
    im1 = preprocess_numpy_image(sensor_data['im_reference'])
    im2 = preprocess_numpy_image(im_comp)

    im1 = torch.rand([1, 1, 128, 128], generator=torch.manual_seed(42))
    im2 = im1 * 0.75
    im1 = torch.nn.functional.interpolate(im1, 256)
    im2 = torch.nn.functional.interpolate(im2, 256)
    print(mssim(im1, im2))
