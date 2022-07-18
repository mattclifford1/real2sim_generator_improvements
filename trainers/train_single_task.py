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


class trainer():
    def __init__(self, dataset_train,
                       dataset_val,
                       model,
                       discrim=None,
                       save_dir='models',
                       batch_size=64,
                       lr=0.0002,
                       lr_decay=0.1,
                       epochs=100,
                       shuffle_train=True,
                       shuffle_val=False):
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.model = model
        self.discriminator = discrim
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.epochs = epochs
        # misc inits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cores = multiprocessing.cpu_count()
        print('Using ', self.cores, ' CPU cores')
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
                                     batch_size=max(1, int(self.batch_size/4)),
                                     shuffle=self.shuffle_val,
                                     num_workers=cores,
                                     prefetch_factor=prefetch_factor)

    def setup(self):
        # optimser
        self.optimiser_generator = optim.Adam(self.model.parameters(), self.lr, betas=(0.5, 0.999))
        self.scheduler = StepLR(self.optimiser_generator, step_size=1, gamma=self.lr_decay)
        if self.discriminator is not None:
            # optimser for the discriminator
            self.optimiser_discrim = optim.Adam(self.discriminator.parameters(), self.lr, betas=(0.5, 0.999))
            self.scheduler_discrim = StepLR(self.optimiser_discrim, step_size=1, gamma=self.lr_decay)
        # self.scheduler = ExponentialLR(self.optimiser, gamma=self.lr_decay)
        # loss criterions for training signals
        self.loss_mse = nn.MSELoss()
        self.model.train()
        self.loss_pixel_wise  = nn.L1Loss()
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(self.device)
            self.discriminator.train()
            # gan training signals
            self.loss_gan = nn.MSELoss()

        self.running_loss = [0]
        self.ssim = 0
        self.MSE = 1*128*128
        self.current_training_loss = 1*128*128
        # attributes to step the learning rate scheduler
        self.step_lr = [0.95, 0.98, 0.99, 0.999, 1]
        self.step_epochs = [50, 100, self.epochs]
        self.step_num = 0
        # set up model for training
        self.model = self.model.to(self.device)

    def get_saver(self):
        # set up save logger for training graphs etc
        if self.discriminator is not None:
            name = 'GAN_'
        else:
            name = 'no_gan'
        self.saver = train_saver(self.save_dir,
                                 self.model,
                                 self.lr,
                                 self.lr_decay,
                                 self.batch_size,
                                 self.dataset_train.task,
                                 name,
                                 self.dataset_train.use_percentage_of_data)

    def start(self, val_every=1, save_model_every=100):
        self.setup()
        self.get_saver()
        self.val_every = val_every
        self.save_model_every = save_model_every
        self.val_all(epoch=0)
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            self.epoch = epoch
            self.running_loss = []
            for step, sample in enumerate(tqdm(self.torch_dataloader_train, desc="Train Steps", leave=False)):
                self.train_step(sample)
            if self.epoch%self.val_every == 0:
                self.val_all(self.epoch+1)
            # self.check_to_lower_learning_rate()
            # if self.ssim > 0.9999:
            #     break

        # # training finished
        self.saver.save_model(self.model, 'final_generator')

    def train_step(self, sample):
        # get training batch sample
        im_real = sample['real'].to(device=self.device, dtype=torch.float)
        im_sim = sample['sim'].to(device=self.device, dtype=torch.float)
        if self.discriminator is not None:
            gan_gt_real = torch.ones(im_real.size(0), 1, self.discriminator.out_size, self.discriminator.out_size).to(device=self.device, dtype=torch.float)
            gan_gt_fake = torch.zeros(im_real.size(0), 1, self.discriminator.out_size, self.discriminator.out_size).to(device=self.device, dtype=torch.float)
        # zero the parameter gradients
        self.optimiser_generator.zero_grad()
        # forward
        pred_sim = self.model(im_real)
        # loss
        loss_generator = 100.0 * self.loss_pixel_wise(pred_sim, im_sim)   # times 100 to copy alex's code with the weighting with discriminator loss
        if self.discriminator is not None:
            pred_fake = self.discriminator(pred_sim, im_real)
            loss_generator += self.loss_gan(pred_fake, gan_gt_real)
        # else:
        #     loss += self.loss_mse(pred_sim, im_sim)_generator
        # backward pass
        loss_generator.backward()
        self.optimiser_generator.step()
        self.running_loss.append(loss_generator.cpu().detach().numpy()) # save the loss stats

        # train discriminator
        if self.discriminator is not None:
            self.optimiser_discrim.zero_grad()
            pred_real = self.discriminator(im_sim, im_real)
            loss_real = self.loss_gan(pred_real, gan_gt_real)
            pred_fake = self.discriminator(pred_sim.detach(), im_real)
            loss_fake = self.loss_gan(pred_fake, gan_gt_fake)

            loss_discriminator = 0.5 * (loss_real + loss_fake)
            loss_discriminator.backward()
            self.optimiser_discrim.step()

    def val_all(self, epoch):
        self.previous_training_loss = self.current_training_loss
        self.current_training_loss = np.mean(self.running_loss)
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

        self.model.train()
        self.MSE = sum(MSEs) / len(MSEs)
        self.ssim = sum(SSIMs) / len(SSIMs)
        stats = {'epoch': [epoch],
                 'mean training loss': [np.mean(self.running_loss)],
                 'val MSE': [self.MSE],
                 'val_SSIM': [self.ssim]}
        self.saver.log_training_stats(stats)
        self.saver.log_val_images(ims, epoch)
        self.maybe_save_model()

    def maybe_save_model(self):
        if self.previous_training_loss > self.current_training_loss:
            self.saver.save_model(self.model, 'best_generator')
        # legacy code to save every x epochs
        # if epoch%self.save_model_every == 0:
        #         # save the trained model
        #         self.saver.save_model(self.model, epoch+1)
        # lower optimiser learning rate

    def check_to_lower_learning_rate(self):
        ## TODO: change to LOSS
        if self.ssim > self.step_lr[self.step_num]:  # if we are scoring well
            self.scheduler.step()
            self.step_num += 1
            print('Learning rate: ', self.scheduler.get_lr())
        # lower if training for many epochs and no improvement on SSIM
        if self.epoch > self.step_epochs[min(len(self.step_epochs)-1, self.step_num)]:
            self.scheduler.step()
            print('Learning rate: ', self.scheduler.get_lr())


if __name__ == '__main__':
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    parser.add_argument("--epochs", type=int, default=250, help='number of epochs to train for')
    parser.add_argument("--task", type=str, nargs='+', default=['edge_2d', 'tap'], help='dataset to train on')
    parser.add_argument("--batch_size",type=int,  default=64, help='batch size to load and train on')
    parser.add_argument("--data_size",type=float,  default=None, help='proportion of training data to use - value between 0 and 1')
    parser.add_argument("--lr",type=float,  default=0.0002, help='learning rate for optimiser')
    parser.add_argument("--pretrained_model", default=False, help='path to model to load pretrained weights on')
    parser.add_argument("--pretrained_name", default='test', help='name to refer to the pretrained model')
    parser.add_argument("--multi_GPU", default=False, action='store_true', help='run on multiple gpus if available')
    parser.add_argument("--GAN", default=False, action='store_true', help='train against discriminator')
    parser.add_argument("--ram", default=False, action='store_true', help='load dataset into ram')
    ARGS = parser.parse_args()

    print('training on: ', ARGS.task)
    dataset_train = image_loader(base_dir=ARGS.dir, task=ARGS.task, store_ram=ARGS.ram, use_percentage_of_data=ARGS.data_size)
    dataset_val = image_loader(base_dir=ARGS.dir, val=True, task=ARGS.task, store_ram=ARGS.ram)
    generator = GeneratorUNet(in_channels=1, out_channels=1)
    if ARGS.GAN == True:
        discriminator = Discriminator(in_channels=1)
    else:
        discriminator = None

    if torch.cuda.device_count() > 1 and ARGS.multi_GPU:
        print("Using ", torch.cuda.device_count(), "GPUs")
        generator = MyDataParallel(generator)
        if ARGS.GAN == True:
            discriminator = MyDataParallel(discriminator)

    if ARGS.pretrained_model == False:
        generator.apply(weights_init_normal)
        if ARGS.GAN == True:
            discriminator.apply(weights_init_normal)
    else:
        weights_init_pretrained(generator, ARGS.pretrained_model, name=ARGS.pretrained_name)

    train = trainer(dataset_train,
                    dataset_val,
                    generator,
                    discriminator,
                    save_dir=os.path.join(ARGS.dir, 'models', 'sim2real', 'matt'),
                    batch_size=ARGS.batch_size,
                    epochs=ARGS.epochs,
                    lr=ARGS.lr)
    train.start()
