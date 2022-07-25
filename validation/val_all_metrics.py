import os
from argparse import ArgumentParser
import gc
import sys; sys.path.append('..'); sys.path.append('.')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import multiprocessing
from torchmetrics.functional import structural_similarity_index_measure as SSIM
# from torchmetrics import StructuralSimilarityIndexMeasure as SSIM   # get GPU memory overflow if using this version.... :/
from torchmetrics.functional import multiscale_structural_similarity_index_measure as MSSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
# from torchmetrics import UniversalImageQualityIndex as UIQI
from torchmetrics.functional import universal_image_quality_index as UIQI
from torchmetrics import SpectralDistortionIndex as SDI
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from trainers.data_loader import image_handler as image_loader
from trainers.utils import show_example_pred_ims
from gan_models.models_128 import GeneratorUNet, weights_init_normal, weights_init_pretrained
from downstream_task.evaller import evaller
from expert.pyramids import LaplacianPyramid

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



class validater():
    def __init__(self,
                dataset_val,
                model,
                downstream_eval=None,
                batch_size=64,
                shuffle_val=False,
                show_ims=False):
        self.dataset_val = dataset_val
        self.shuffle_val = shuffle_val
        self.model = model
        self.downstream_eval = downstream_eval
        self.batch_size = batch_size
        self.show_ims = show_ims
        # misc inits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cores = multiprocessing.cpu_count()
        # get data loader
        self.get_data_loader(prefetch_factor=1)
        self.get_similarity_metrics()

    def get_data_loader(self, prefetch_factor=1):
        cores = max(1, int(self.cores/2))
        self.torch_dataloader_val = DataLoader(self.dataset_val,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle_val,
                                     num_workers=cores,
                                     prefetch_factor=prefetch_factor)

    def get_similarity_metrics(self):
        # dictionary of callable metric functions
        # eg: score = self.metrics['xx'](im1, im2)
        self.metrics = {}
        self.metrics['MSE'] = nn.MSELoss()
        self.metrics['SSIM'] = SSIM
        self.metrics['NLPD'] = LaplacianPyramid(k=1).to(self.device)
        # self.metrics['MSSIM'] = MSSIM    # not compatable with 128x128 images
        self.metrics['PSNR'] = PSNR().to(self.device)
        self.metrics['UIQI'] = UIQI
        # self.metrics['SDI'] = SDI() # gives nan
        # self.metrics['LPIPS_alex'] = grey_to_3_channel_input(LPIPS(net_type='alex').to(self.device))
        self.metrics['LPIPS_vgg'] = grey_to_3_channel_input(LPIPS(net_type='vgg').to(self.device))
        # self.metrics['NLPD_2'] = LaplacianPyramid(k=2).to(self.device)
        # self.metrics['NLPD_3'] = LaplacianPyramid(k=3).to(self.device)

    def start(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        # ims_to_save = 5
        # ims = []
        # initialise lists to hold metric's scores
        _scores = {}
        for key in self.metrics.keys():
            _scores[key] = []
        for step, sample in enumerate(tqdm(self.torch_dataloader_val, desc="Val Steps", leave=False)):
            # get val batch sample
            im_real = sample['real'].to(device=self.device, dtype=torch.float)
            im_sim = sample['sim'].to(device=self.device, dtype=torch.float)
            # forward
            pred_sim = self.model(im_real)
            # get metrics
            for key in self.metrics.keys():
                _score = self.metrics[key](im_sim, pred_sim)
                # _score = self.metrics[key](im_sim.cpu(), pred_sim.cpu())
                _scores[key].append(_score.cpu().detach().numpy())

            # store some ims to save to inspection
            # if len(ims) < ims_to_save:
            #     ims.append({'predicted': pred_sim[0,0,:,:],
            #                 'simulated': im_sim[0,0,:,:],
            #                 'real': im_real[0,0,:,:]})
            # elif self.show_ims == True:
            #     show_example_pred_ims(ims)
            #     ims = []

            # uncomment below when developing code
            # if step == 1:
            #     break


        # Calculate mean of scores
        stats = {}
        for key in self.metrics.keys():
            stats[key] = sum(_scores[key]) / len(_scores[key])

        # evaluate on downstream task (outside of main loops as requires a diff data loader to get y labels)
        if self.downstream_eval is not None:
             stats['Downstream MAE'] =  self.downstream_eval.get_MAE(self.model)

        self.stats = stats
        # self.saver.log_training_stats(stats)
        # self.saver.log_val_images(ims, epoch)

def run_val(dir='..',
            task=['edge_2d', 'tap'],
            batch_size=32,
            pretrained_model=False,
            pretrained_name='test',
            ram=False,
            show_ims=False):
    dataset_val = image_loader(base_dir=dir, val=True, task=task, store_ram=ram)
    generator = GeneratorUNet(in_channels=1, out_channels=1)

    if pretrained_model == False:
        generator.apply(weights_init_normal)
    else:
        weights_init_pretrained(generator, pretrained_model, name=pretrained_name)

    downstream_eval = evaller(dir, data_task=task+['real'],
                                   model_task=task+['sim'],
                                   run=0,
                                   store_ram=ram,
                                   batch_size=max(1, int(batch_size/4)))

    v = validater(dataset_val,
                    generator,
                    downstream_eval,
                    batch_size=max(1, int(batch_size/4)),
                    show_ims=show_ims)
    v.start()
    stats = v.stats

    # avoid memory overflow by clearing VRAM
    if torch.cuda.is_available():
        del v
        gc.collect()
        torch.cuda.empty_cache()

    return stats


if __name__ == '__main__':
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    parser.add_argument("--task", type=str, nargs='+', default=['surface_3d', 'shear'], help='dataset to train on')
    parser.add_argument("--batch_size",type=int,  default=32, help='batch size to load and train on')
    parser.add_argument("--pretrained_model", default=False, help='path to model to load pretrained weights on')
    parser.add_argument("--pretrained_name", default='test', help='name to refer to the pretrained model')
    parser.add_argument("--ram", default=False, action='store_true', help='load dataset into ram')
    parser.add_argument("--show_ims", default=False, action='store_true', help='show some example images')
    ARGS = parser.parse_args()
    stats = run_val(dir=ARGS.dir,
                task=ARGS.task,
                batch_size=ARGS.batch_size,
                pretrained_model=ARGS.pretrained_model,
                pretrained_name=ARGS.pretrained_name,
                ram=ARGS.ram,
                show_ims=ARGS.show_ims)
    for key in stats.keys():
        print(key,': ',stats[key])
