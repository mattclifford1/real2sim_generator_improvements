'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

evaluation of tactip pose estimation nueral network
'''
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys; sys.path.append('..'); sys.path.append('.')
import downstream_task.networks.model_128 as m_128
from downstream_task.data import dataloader
from downstream_task.pose_net_utils import load_weights


class evaller:
    def __init__(self, dir='..',
                       data_task=('surface_3d', 'shear', 'real'),
                       model_task=('surface_3d', 'shear', 'real'),
                       train_name='_LR:0.0001_BS:16',
                       run=0,
                       model='best_model.pth',
                       store_ram=False,
                       batch_size=4):
        self.dir = dir    # base dir where all models are held
        self.data_task = data_task  # data_task to load eg. ['surface_3d', 'shear']
        self.model_task = model_task  # model_task to load eg. ['surface_3d', 'shear']
        self.train_name = train_name
        self.run = 'run_' + str(run)
        self.model_dir = os.path.join(self.dir, 'models', 'pose_estimation', self.model_task[0], self.model_task[1], self.model_task[2]+self.train_name, self.run, 'checkpoints')
        self.model_name = model
        self.store_ram = store_ram
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_pretrained_model()
        self.get_val_data()

    def get_val_data(self):
        validation_data = dataloader.get_data(self.dir, self.data_task, store_ram=self.store_ram, val=True, labels_range=self.normalisation)
        self.data_loader = DataLoader(validation_data, self.batch_size)


    def get_pretrained_model(self):
        self.model = m_128.network(final_size=int(self.model_task[0][-2]), task=self.model_task[0])
        model_file = os.path.join(self.model_dir, self.model_name)
        load_weights(self.model, model_file)
        self.model.to(self.device)
        normalisation_file = os.path.join(self.model_dir, 'output_normilisation.json')
        with open(normalisation_file) as f:
            self.normalisation = json.load(f)

    def predict(self, ims):
        # ims should be a torch tensor on the correct device
        preds = self.model(ims)

    def _get_MAE_batch(self, ims, labels):
        preds = self.model(ims)
        mae = torch.abs(preds - labels).mean()
        mae = mae.cpu().detach().numpy()
        return mae

    def get_MAE(self, real2sim_model=None):
        MAEs = []
        for step, sample in enumerate(tqdm(self.data_loader, desc="Downstream Val Steps", leave=False)):
            # get val batch sample
            im = sample['image'].to(device=self.device, dtype=torch.float)
            label = sample['label'].to(device=self.device, dtype=torch.float)
            if real2sim_model is not None:
                # shift domain from real to simulation
                im = real2sim_model(im)
            MAEs.append(self._get_MAE_batch(im, label))
            # uncomment below when developing code
            # if step == 1:
            #     break
        return sum(MAEs) / len(MAEs)




if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--dir", default='..', type=str, help='folder where data is located')
    parser.add_argument("--data_task", type=str, nargs='+', default=['surface_3d', 'shear', 'sim'], help='dataset to eval on')
    parser.add_argument("--model_task", type=str, nargs='+', default=['surface_3d', 'shear', 'sim'], help='model to eval on')
    parser.add_argument("--batch_size",type=int,  default=16, help='batch size to load and train on')
    parser.add_argument("--ram", default=False, action='store_true', help='load dataset into ram')
    ARGS = parser.parse_args()

    # e = evaller(ARGS.dir, data_task=ARGS.data_task, model_task=ARGS.model_task)
    #
    # mae =e.get_MAE()
    # print(mae)
    import itertools
    import pandas as pd
    task = ['edge_2d','surface_3d']
    sampling = ['tap', 'shear']
    # domain = ['real', 'sim']
    domain = ['real']
    posenets = list(itertools.product(task, sampling, domain))

    posenets = [('surface_3d', 'shear')]
    from gan_models.models_128 import GeneratorUNet, weights_init_pretrained
    generator = GeneratorUNet(in_channels=1, out_channels=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {'run':[0]}
    results = {'run':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    results = {'run':[0, 1, 2]}
    for net in tqdm(posenets, desc='posenets'):
        results[str(net)] = []
        for run in tqdm(results['run'], desc='runs', leave=False):
            # e = evaller(ARGS.dir, data_task=net, model_task=net, run=run)
            gen_model = os.path.join(ARGS.dir, 'models/sim2real/alex/trained_gans/['+net[0]+']/128x128_['+net[1]+']_250epochs', 'checkpoints', 'best_generator.pth')
            gen_model = os.path.join(ARGS.dir, 'models/sim2real/matt/'+net[0]+'/'+net[1], 'not_pretrained', 'GAN_LR:0.0002_decay:0.1_BS:64_DS:1.0', 'run_'+str(run), 'models', 'final_generator.pth')
            weights_init_pretrained(generator, gen_model)
            generator = generator.to(device)
            generator.eval()

            e = evaller(ARGS.dir, data_task=('surface_3d', 'shear', 'real'), model_task=('surface_3d', 'shear', 'sim'), run=run)
            mae = e.get_MAE(generator)
            results[str(net)].append(mae)

        df = pd.DataFrame.from_dict(results)
        # df.to_csv('downstream_task/validation_results.csv')
        df.to_csv('downstream_task/validation_results_diff_runs.csv')
