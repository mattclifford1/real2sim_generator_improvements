'''
Script to train UNet with adverserial training.
Code adapted from Alex Churh's tactile_gym_sim2real repo

Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk
'''
import os
from argparse import ArgumentParser
import sys; sys.path.append('..'); sys.path.append('.')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR, StepLR

from tqdm import tqdm
import multiprocessing
from torchmetrics.functional import structural_similarity_index_measure as SSIM

from trainers.data_loader import image_handler as image_loader
from trainers.utils import train_saver, MyDataParallel
from gan_models.models_128 import GeneratorUNet, Discriminator, weights_init_normal, weights_init_pretrained


class validater():
    def __init__(self,
                dataset_val,
                model,
                batch_size=64,
                shuffle_val=False):
        self.dataset_val = dataset_val
        self.shuffle_val = shuffle_val
        self.model = model
        self.batch_size = batch_size
        # misc inits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cores = multiprocessing.cpu_count()
        # get data loader
        self.get_data_loader(prefetch_factor=1)

    def get_data_loader(self, prefetch_factor=1):
        cores = int(self.cores/2)
        self.torch_dataloader_val = DataLoader(self.dataset_val,
                                     batch_size=max(1, int(self.batch_size/4)),
                                     shuffle=self.shuffle_val,
                                     num_workers=cores,
                                     prefetch_factor=prefetch_factor)

    def start(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        self.val_all()

    def val_all(self):
        self.model.eval()
        MSEs = []
        SSIMs = []
        ims_to_save = 3
        ims = []
        for step, sample in enumerate(tqdm(self.torch_dataloader_val, desc="Val Steps", leave=False)):
            # get val batch sample
            im_real = sample['real'].to(device=self.device, dtype=torch.float)
            im_sim = sample['sim'].to(device=self.device, dtype=torch.float)
            # forward
            pred_sim = self.model(im_real)
            # get metrics
            mse = torch.square(pred_sim - im_sim).mean()
            MSEs.append(mse.cpu().detach().numpy())
            ssim = SSIM(pred_sim, im_sim)
            SSIMs.append(ssim.cpu().detach().numpy())

            # store some ims to save to inspection
            if len(ims) < ims_to_save:
                ims.append({'predicted': pred_sim[0,0,:,:],
                            'simulated': im_sim[0,0,:,:],
                            'real': im_real[0,0,:,:]})

        self.MSE = sum(MSEs) / len(MSEs)
        self.ssim = sum(SSIMs) / len(SSIMs)
        stats = {'val MSE': [self.MSE],
                 'val_SSIM': [self.ssim]}

        for key in stats.keys():
            print(key,': ',stats[key])


        # self.saver.log_training_stats(stats)
        # self.saver.log_val_images(ims, epoch)

def run_val(dir='..',
            task=['edge_2d', 'tap'],
            batch_size=64,
            pretrained_model=False,
            pretrained_name='test',
            ram=False):
    print('validating on: ', ARGS.task)
    dataset_val = image_loader(base_dir=ARGS.dir, val=True, task=ARGS.task, store_ram=ARGS.ram)
    generator = GeneratorUNet(in_channels=1, out_channels=1)

    if ARGS.pretrained_model == False:
        generator.apply(weights_init_normal)
    else:
        weights_init_pretrained(generator, ARGS.pretrained_model, name=ARGS.pretrained_name)

    v = validater(dataset_val,
                    generator,
                    batch_size=ARGS.batch_size)
    v.start()


if __name__ == '__main__':
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    parser.add_argument("--task", type=str, nargs='+', default=['edge_2d', 'tap'], help='dataset to train on')
    parser.add_argument("--batch_size",type=int,  default=64, help='batch size to load and train on')
    parser.add_argument("--pretrained_model", default=False, help='path to model to load pretrained weights on')
    parser.add_argument("--pretrained_name", default='test', help='name to refer to the pretrained model')
    parser.add_argument("--ram", default=False, action='store_true', help='load dataset into ram')
    ARGS = parser.parse_args()
    run_val(dir=ARGS.dir,
                task=ARGS.task,
                batch_size=ARGS.batch_size,
                pretrained_model=ARGS.pretrained_model,
                pretrained_name=ARGS.pretrained_name,
                ram=ARGS.ram)
