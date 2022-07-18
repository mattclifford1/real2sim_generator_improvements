'''
Load sim and real data with torch data loader

Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk
'''
import os
from argparse import ArgumentParser
import cv2
import torch
from tqdm import tqdm
import random
import sys; sys.path.append('..'); sys.path.append('.')
from trainers.image_transforms import process_image, process_image_sim
from trainers.utils import check_task

def get_all_ims(dir, ext='.png'):
    ims = []
    for im in os.listdir(dir):
        if os.path.splitext(im)[1] == ext:
            ims.append(im)
    return ims

def get_params(val=False):
    if not val:
        return {
                  # 'dims':        (256, 256),
                  'rshift':      (0.025, 0.025),
                  'rzoom':       None,  # (0.98, 1),
                  'thresh':      True,
                  'brightlims':  None,  # [0.3, 1.0, -50, 50], # alpha limits for contrast, beta limits for brightness
                  'noise_var':   None,  # 0.001,
                  'stdiz':       False,
                  'normlz':      True,
                  'joint_aug':   False,
                  'bbox':        [80,25,530,475],
                  'gray':        True,
                  'add_axis':    False
                  }
    else:
        return {
                  # 'dims':        (256, 256),
                  'rshift':      None,
                  'rzoom':       None,  # (0.98, 1),
                  'thresh':      True,
                  'brightlims':  None,  # [0.3, 1.0, -50, 50], # alpha limits for contrast, beta limits for brightness
                  'noise_var':   None,  # 0.001,
                  'stdiz':       False,
                  'normlz':      True,
                  'joint_aug':   False,
                  'bbox':        [80,25,530,475],
                  'gray':        True,
                  'add_axis':    False
                  }

def numpy_image_to_torch_tensor(sample):
    """Convert ndarrays in sample to Tensors."""
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C x H x W
    for key in sample.keys():
        # add chanel dim to 2D arrays
        if len(sample[key].shape) == 2:
            sample[key] = np.expand_dims(sample[key], axis=2)
        sample[key] = torch.from_numpy(sample[key].transpose((2, 0, 1)))
    return sample


class image_handler():
    def __init__(self,
                 base_dir='..',
                 data='data/Bourne/tactip',
                 task=('edge_2d', 'tap'),
                 size=128,
                 val=False,
                 store_ram=False,
                 use_percentage_of_data=None):
        # real_images_dir = os.path.join(dir, 'data/Bourne/tactip/real/'+data[0]+'/'+data[1]+'/csv_val/images')
        check_task(task)
        self.dir = os.path.join(base_dir, data)
        self.task = task
        self.split_type = 'csv_val' if val else 'csv_train'
        self.real_dir = os.path.join(self.dir, 'real', task[0], task[1], self.split_type, 'images')
        self.sim_dir = os.path.join(self.dir, 'sim', task[0], task[1], str(size)+'x'+str(size), self.split_type, 'images')
        self.images = get_all_ims(self.real_dir)
        self.check_image_pairs_exist()
        self.im_params = get_params(val)
        self.im_params['size'] = (size, size)
        self.use_percentage_of_data = use_percentage_of_data
        if self.use_percentage_of_data is not None:
            self.prune_data()
        self.store_ram = store_ram
        if self.store_ram == True:
            self.load_dataset_to_ram()

    def check_image_pairs_exist(self):
        for image in self.images:
            sim_im = os.path.join(self.sim_dir, image)
            if not os.path.isfile(sim_im):
                raise Exception(sim_im, ' image not found')

    def prune_data(self):
        ''' only use a proportion of the data, takes a random sample of the data'''
        if self.use_percentage_of_data < 0 or self.use_percentage_of_data > 1:
            raise Exception('use_percentage_of_data needs to be between 0 and 1')
        random.shuffle(self.images)
        to_use = max(1, int(len(self.images)*self.use_percentage_of_data))
        self.images = self.images[:to_use]

    def load_dataset_to_ram(self):
        self.data_in_ram = {'real':[], 'sim':[]}
        for i in tqdm(range(len(self.images)), desc='Loading dataset into RAM', leave=False):
            real_image_filename = os.path.join(self.real_dir, self.images[i])
            sim_image_filename = os.path.join(self.sim_dir, self.images[i])
            # print('=======================')
            # print(real_image_filename)
            self.data_in_ram['real'].append(cv2.imread(real_image_filename))
            # print(sim_image_filename)
            self.data_in_ram['sim'].append(cv2.imread(sim_image_filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        if self.store_ram == True:  # get pre loaded from ram
            raw_real_image = self.data_in_ram['real'][i]
            raw_sim_image = self.data_in_ram['sim'][i]
        else:   # load from disk
            # get image filepaths
            real_image_filename = os.path.join(self.real_dir, self.images[i])
            sim_image_filename = os.path.join(self.sim_dir, self.images[i])
            # load images from filename
            raw_real_image = cv2.imread(real_image_filename)
            raw_sim_image = cv2.imread(sim_image_filename)
        # preprocess images
        processed_real_image = process_image(raw_real_image, self.im_params)
        processed_sim_image = process_image_sim(raw_sim_image, self.im_params)
        # create sample and convert to torch
        sample = {"real": processed_real_image, "sim": processed_sim_image}
        sample = numpy_image_to_torch_tensor(sample)
        return sample


if __name__ == '__main__':
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    ARGS = parser.parse_args()

    h = image_handler(base_dir=ARGS.dir)
    print(h[0])
