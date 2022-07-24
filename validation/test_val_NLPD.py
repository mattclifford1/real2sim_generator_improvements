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
from trainers.utils import train_saver, MyDataParallel, show_example_pred_ims
from gan_models.models_128 import GeneratorUNet, Discriminator, weights_init_normal, weights_init_pretrained
from downstream_task.evaller import evaller
from expert.pyramids import LaplacianPyramid


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
        self.NLPD = LaplacianPyramid(k=1)
        self.NLPD.to(self.device)
        self.MSEloss = nn.MSELoss()

    def get_data_loader(self, prefetch_factor=1):
        cores = int(self.cores/2)
        self.torch_dataloader_val = DataLoader(self.dataset_val,
                                     batch_size=self.batch_size,
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
        MSE_losses = []
        SSIMs = []
        NLPDs = []
        ims_to_save = 5
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
            mse_loss = self.MSEloss(pred_sim, im_sim)
            MSE_losses.append(mse_loss.cpu().detach().numpy())
            ssim = SSIM(pred_sim, im_sim)
            SSIMs.append(ssim.cpu().detach().numpy())
            # NLPD
            pred_sim_3 = torch.cat([pred_sim, pred_sim, pred_sim], dim=1)
            im_sim_3 = torch.cat([im_sim, im_sim, im_sim], dim=1)
            nlpd = self.NLPD.compare(pred_sim_3, im_sim_3)
            NLPDs.append(nlpd.cpu().detach().numpy())
            # store some ims to save to inspection
            if len(ims) < ims_to_save:
                ims.append({'predicted': pred_sim[0,0,:,:],
                            'simulated': im_sim[0,0,:,:],
                            'real': im_real[0,0,:,:]})
            elif self.show_ims == True:
                show_example_pred_ims(ims)
                ims = []
            # if step == 3:
            #     break

        self.MSE = sum(MSEs) / len(MSEs)
        self.MSE_loss = sum(MSE_losses) / len(MSE_losses)
        self.NLPD = sum(NLPDs) / len(MSE_losses)
        self.ssim = sum(SSIMs) / len(NLPDs)
        stats = {'val MSE': [self.MSE],
                 'val MSE loss': [self.MSE_loss],
                 'val NLPD': [self.NLPD],
                 'val_SSIM': [self.ssim],
                 }
        if self.downstream_eval is not None:
             stats['Downstream MAE'] =  [self.downstream_eval.get_MAE(self.model)]

        for key in stats.keys():
            print(key,': ',stats[key])

        # self.saver.log_training_stats(stats)
        # self.saver.log_val_images(ims, epoch)

def run_val(dir='..',
            task=['edge_2d', 'tap'],
            batch_size=64,
            pretrained_model=False,
            pretrained_name='test',
            ram=False,
            show_ims=False):
    print('validating on: ', task)
    dataset_val = image_loader(base_dir=dir, val=True, task=task, store_ram=ram)
    generator = GeneratorUNet(in_channels=1, out_channels=1)

    if ARGS.pretrained_model == False:
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


if __name__ == '__main__':
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    parser.add_argument("--task", type=str, nargs='+', default=['surface_3d', 'shear'], help='dataset to train on')
    parser.add_argument("--batch_size",type=int,  default=64, help='batch size to load and train on')
    parser.add_argument("--pretrained_model", default=False, help='path to model to load pretrained weights on')
    parser.add_argument("--pretrained_name", default='test', help='name to refer to the pretrained model')
    parser.add_argument("--ram", default=False, action='store_true', help='load dataset into ram')
    parser.add_argument("--show_ims", default=False, action='store_true', help='show some example images')
    ARGS = parser.parse_args()
    run_val(dir=ARGS.dir,
                task=ARGS.task,
                batch_size=ARGS.batch_size,
                pretrained_model=ARGS.pretrained_model,
                pretrained_name=ARGS.pretrained_name,
                ram=ARGS.ram,
                show_ims=ARGS.show_ims)
