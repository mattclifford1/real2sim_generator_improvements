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

def check_task(task_tuple):
    task0 = ['surface_3d', 'edge_2d']
    task1 = ['tap', 'shear']
    if len(task_tuple) != 2:
        raise Exception('Task needs to be length 2')
    if task_tuple[0] not in task0:
        raise Exception('first task arg needs to be either: ', task0, ' not:', str(task_tuple[0]))
    if task_tuple[1] not in task1:
        raise Exception('second task arg needs to be either: ', task1, ' not:', str(task_tuple[1]))

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
    def __init__(self, base_dir,
                       model,
                       lr,
                       lr_decay,
                       batch_size,
                       task,
                       save_name=''):
        self.base_dir = base_dir
        self.task = task
        self.save_name = save_name
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
        name = self.save_name
        name = name + 'LR:'+str(self.lr)
        name = name +'_decay:'+str(self.lr_decay)
        name = name +'_BS:'+str(self.batch_size)
        self.dir = os.path.join(dir, name)
        # find if there are previous runs
        run_name = 'run_'
        if os.path.isdir(self.dir):
            # shutil.rmtree(self.dir)
            runs = [int(i[len(run_name):]) for i in os.listdir(self.dir)]
            run_num = max(runs) + 1
        else:
            run_num = 0
        self.dir = os.path.join(self.dir, run_name+str(run_num))
        self.models_dir = os.path.join(self.dir, 'models')
        self.ims_dir = os.path.join(self.dir, 'ims')
        # make dirs is dont already exist
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if not os.path.exists(self.ims_dir):
            os.makedirs(self.ims_dir)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def load_pretrained(self, model):
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

    def save_model(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.models_dir, str(name)+'.pth'))

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


def show_example_pred_ims(ims):
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
    plt.show()
