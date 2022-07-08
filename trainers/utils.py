'''
Train helper functions

Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk
'''
import os
import shutil
import torch
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)   # torch warning we dont care about

class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class train_saver:
    def __init__(self, base_dir, model, lr, lr_decay, batch_size, task, from_scratch=False):
        self.base_dir = base_dir
        self.task = task
        self.from_scratch = from_scratch
        if hasattr(model, 'name'):
            self.model_name = model.name
        else:
            self.model_name = model.__class__.__name__
        self.pretrained_name = model.init_weights_from
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.get_save_dir()

    def get_save_dir(self):
        dir = os.path.join(self.base_dir, self.task[0], self.task[1], self.pretrained_name)
        name = 'LR:'+str(self.lr)
        # name = name +'_decay:'+str(self.lr_decay)
        name = name +'_BS:'+str(self.batch_size)
        self.dir = os.path.join(dir, name)
        if self.from_scratch and os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
        self.models_dir = os.path.join(self.dir, 'models')
        self.ims_dir = os.path.join(self.dir, 'ims')
        # make dirs is dont already exist
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if not os.path.exists(self.ims_dir):
            os.makedirs(self.ims_dir)

    def load_pretrained(self, model):
        if not os.path.isdir(self.models_dir):
            os.mkdir(self.models_dir)
            return 0
        checkpoints = os.listdir(self.models_dir)
        saves = []
        for checkpoint in checkpoints:
            name, ext = os.path.splitext(checkpoint)
            try:
                epoch = int(name)
                saves.append(epoch)
            except:
                pass
        if len(saves) > 0:
            latest_epoch = max(saves)
            weights_path = os.path.join(self.models_dir, str(latest_epoch)+'.pth')
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            print('Loaded previously trained model at epoch: '+str(latest_epoch))
            return latest_epoch
        else:
            return 0 #no pretrained found

    def save_model(self, model, epoch):
        torch.save(model.state_dict(), os.path.join(self.models_dir, str(epoch)+'.pth'))

    def log_training_stats(self, stats_dicts):
        df = pd.DataFrame(stats_dicts)
        # load csv if there is one
        file = os.path.join(self.dir, 'training_stats.csv')
        if os.path.isfile(file):
            df.to_csv(file, mode='a', index=False, header=False)
        else:
            df.to_csv(file, index=False)

    def log_val_images(self, ims, epoch):
        f, axarr = plt.subplots(nrows=len(ims), ncols=3)
        axarr[0,0].set_title('real')
        axarr[0,1].set_title('predicted')
        axarr[0,2].set_title('simulated')
        for i, im_dict in enumerate(ims):
            axarr[i,0].imshow(im_dict['real'].cpu().detach().numpy(), cmap='gray')
            axarr[i,1].imshow(im_dict['predicted'].cpu().detach().numpy(), cmap='gray')
            axarr[i,2].imshow(im_dict['simulated'].cpu().detach().numpy(), cmap='gray')
            axarr[i,0].axis('off')
            axarr[i,1].axis('off')
            axarr[i,2].axis('off')
        plt.savefig(os.path.join(self.ims_dir, str(epoch)+'.png'))
        plt.close(f) 
