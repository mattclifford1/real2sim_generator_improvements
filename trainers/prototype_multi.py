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


if __name__ == '__main__':
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    parser.add_argument("--task", type=str, nargs='+', default=['edge_2d', 'tap'], help='dataset to train on')
    ARGS = parser.parse_args()


    dataset_train_1 = image_loader(base_dir=ARGS.dir, task=['edge_2d', 'tap'])
    dataset_train_2 = image_loader(base_dir=ARGS.dir, task=['edge_2d', 'shear'])

    cores = int(multiprocessing.cpu_count()/4)
    l1 = DataLoader(dataset_train_1,
                     batch_size=64,
                     shuffle=False,
                     num_workers=cores,
                     prefetch_factor=1)
    l2 = DataLoader(dataset_train_2,
                     batch_size=2,
                     shuffle=False,
                     num_workers=cores,
                     prefetch_factor=1)

    for step, sample in enumerate(tqdm(l1, desc="Train Steps", leave=False)):
        continue
