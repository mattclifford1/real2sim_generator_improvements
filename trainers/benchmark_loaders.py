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
    parser.add_argument("--batch_size",type=int,  default=64, help='batch size to load and train on')
    parser.add_argument("--prefetch",type=int,  default=1, help='prefetch data amount')
    parser.add_argument("--cores",type=int,  default=int(multiprocessing.cpu_count()/2), help='number of cpu cores to use')
    parser.add_argument("--epochs",type=int,  default=5, help='epochs to use')
    parser.add_argument("--use_cpu", default=False, action='store_true', help='force cpu use')
    parser.add_argument("--ram", default=False, action='store_true', help='load dataset into ram')
    ARGS = parser.parse_args()


    dataset_train_1 = image_loader(base_dir=ARGS.dir,
                                   task=ARGS.task,
                                   store_ram=ARGS.ram)

    cores = int(multiprocessing.cpu_count()/2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ARGS.use_cpu == True:
        device = 'cpu'
    l1 = DataLoader(dataset_train_1,
                     batch_size=ARGS.batch_size,
                     shuffle=False,
                     num_workers=ARGS.cores,
                     prefetch_factor=ARGS.prefetch)

    for epoch in tqdm(range(ARGS.epochs), desc="epochs", leave=True):
        for step, sample in enumerate(tqdm(l1, desc="Data loader", leave=False)):
            im_real = sample['real'].to(device=device, dtype=torch.float)
            im_sim = sample['sim'].to(device=device, dtype=torch.float)
