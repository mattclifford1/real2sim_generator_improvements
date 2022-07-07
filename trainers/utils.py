'''
Train helper functions

Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk
'''
import os
import torch
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)   # torch warning we dont care about

class train_saver:
    def __init__(self, base_dir, model, lr, lr_decay, batch_size):
        self.base_dir = base_dir
        print(self.base_dir)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        if hasattr(model, 'name'):
            self.model_name = model.name
        else:
            self.model_name = model.__class__.__name__
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.get_save_dir()

    def get_save_dir(self):
        dir = os.path.join(self.base_dir, self.model_name)
        dir = dir +'_LR_'+str(self.lr)
        dir = dir +'_decay_'+str(self.lr_decay)
        self.dir = dir +'_BS_'+str(self.batch_size)

    def load_pretrained(self, model):
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
            return 0
        checkpoints = os.listdir(self.dir)
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
            weights_path = os.path.join(self.dir, str(latest_epoch)+'.pth')
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            print('Loaded pretrained model at epoch: '+str(latest_epoch))
            return latest_epoch
        else:
            return 0 #no pretrained found

    def save_model(self, model, epoch):
        torch.save(model.state_dict(), os.path.join(self.dir, str(epoch)+'.pth'))

    def log_training_stats(self, stats_dicts):
        df = pd.DataFrame(stats_dicts)
        # load csv if there is one
        file = os.path.join(self.dir, 'training_stats.csv')
        if os.path.isfile(file):
            df.to_csv(file, mode='a', index=False, header=False)
        else:
            df.to_csv(file, index=False)
