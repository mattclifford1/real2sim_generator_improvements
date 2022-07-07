'''
Script to train UNet with supervised normal signal

Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk
'''
import os
from argparse import ArgumentParser
import sys; sys.path.append('..'); sys.path.append('.')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm
import multiprocessing

from trainers.data_loader import image_handler as image_loader
from gan_models.models_128 import GeneratorUNet, weights_init_normal


class trainer():
    def __init__(self, dataset_train,
                       dataset_val,
                       model,
                       batch_size=64,
                       lr=1e-4,
                       decay=1e-6,
                       epochs=100,
                       shuffle_train=False):
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.shuffle_train = shuffle_train
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.decay = decay
        self.epochs = epochs
        # misc inits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cores = multiprocessing.cpu_count()
        # get data loader
        self.get_data_loader(prefetch_factor=2)

    def get_data_loader(self, prefetch_factor=1):
        cores = int(self.cores/2)
        self.torch_dataloader_train = DataLoader(self.dataset_train,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle_train,
                                     num_workers=cores,
                                     prefetch_factor=prefetch_factor)

        self.torch_dataloader_val = DataLoader(self.dataset_val,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=cores,
                                     prefetch_factor=prefetch_factor)

    def setup(self):
        # optimser
        self.optimiser = optim.Adam(self.model.parameters(), self.lr)
        # self.scheduler = ExponentialLR(self.optimiser, gamma=self.lr_decay)
        # loss criterion for training signal
        self.loss = nn.MSELoss()
        # set up model for training
        self.model = self.model.to(self.device)
        self.model.train()

    def start(self):
        self.setup()
        self.val_every = 1
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            for step, sample in enumerate(tqdm(self.torch_dataloader_train, desc="Train Steps", leave=False)):
                self.train_step(sample)
            if epoch%self.val_every == 0:
                self.val_all()

    def train_step(self, sample):
        # get training batch sample
        im_real = sample['real'].to(device=self.device, dtype=torch.float)
        im_sim = sample['sim'].to(device=self.device, dtype=torch.float)
        # zero the parameter gradients
        self.optimiser.zero_grad()
        # forward
        pred_sim = self.model(im_real)
        # loss
        loss = self.loss(pred_sim, im_sim)
        # backward pass
        loss.backward()
        self.optimiser.step()
        # self.running_loss.append(loss.cpu().detach().numpy()) # save the loss stats

    def val_all(self):
        self.model.eval()
        MSEs = []
        for step, sample in enumerate(tqdm(self.torch_dataloader_val, desc="Val Steps", leave=False)):
            # get val batch sample
            im_real = sample['real'].to(device=self.device, dtype=torch.float)
            im_sim = sample['sim'].to(device=self.device, dtype=torch.float)
            # forward
            pred_sim = self.model(im_real)
            MSEs.append(torch.square(pred_sim - im_sim).mean())
        self.model.train()
        MSE = sum(MSEs) / len(MSEs)
        print(MSE)
        # now save to csv/plot graphs

if __name__ == '__main__':
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    ARGS = parser.parse_args()

    dataset_train = image_loader(base_dir=ARGS.dir)
    dataset_val = image_loader(base_dir=ARGS.dir, val=True)
    generator = GeneratorUNet(in_channels=1, out_channels=1)
    generator.apply(weights_init_normal)

    train = trainer(dataset_train, dataset_val, generator , batch_size=2, epochs=1)
    train.start()
