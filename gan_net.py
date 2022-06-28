import os
import numpy as np
import time

import torch.nn as nn
import torch.nn.functional as F
import torch

from image_transforms import *

import json
def load_json_obj(name):
    with open(name+'.json', 'r') as fp:
        return json.load(fp)


class GAN_io():

    def __init__(self, gan_model_dir, rl_image_size=[64,64]):

        self.rl_image_size = rl_image_size
        self.params = load_json_obj(os.path.join(gan_model_dir, 'augmentation_params'))

        # overide some augmentation params as we dont want them when generating new data
        self.params['bbox'] = [80, 25, 530, 475] # TODO: make sure correct
        self.params['rshift'] = None
        self.params['rzoom'] = None
        self.params['brightlims'] = None
        self.params['noise_var'] = None


        # configure gpu use
        self.cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        # if self.cuda:
        #     self.generator = generator.cuda()
        # else:
        #     self.generator = generator

        # put in eval mode to disable dropout etc
        # self.generator.eval()
        # self.generator = gen

    def process_raw_image(self, im):
        # preprocess/augment image
        processed_real_image = process_image(
            im, gray=True,
            bbox=self.params['bbox'], dims=self.params['dims'],
            stdiz=self.params['stdiz'], normlz=self.params['normlz'],
            rshift=self.params['rshift'], rzoom=self.params['rzoom'],
            thresh=self.params['thresh'], add_axis=False,
            brightlims=self.params['brightlims'], noise_var=self.params['noise_var']
        )
        return processed_real_image


    def to_tensor(self, im):
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=2)
        # put the channel into first axis because pytorch
        im_pt = np.rollaxis(im, 2, 0)

        # add an axis to make a batch
        im_pt = im_pt[np.newaxis, ...]

        # convert to torch tensor
        im_pt = torch.from_numpy(im_pt).type(self.Tensor)
        return im_pt

    def to_numpy(self, gen_sim_image):
        # convert to numpy, image format, size expected by rl agent
        gen_sim_image = gen_sim_image[0,0,...].detach().cpu().numpy() # pytorch batch -> numpy image
        gen_sim_image = (np.clip(gen_sim_image, 0, 1)*255).astype(np.uint8) # convert to image format

        if self.params['dims'] != self.rl_image_size:
            gen_sim_image = cv2.resize(gen_sim_image, tuple(self.rl_image_size), interpolation=cv2.INTER_NEAREST) # resize to RL expected
        return gen_sim_image

    # def gen_sim_image(self, real_image):
    #     # preprocess/augment image
    #     processed_real_image = self.process_raw_image(real_image)
    #
    #     # setup the processed image for plotting
    #     processed_real_image_plot = (np.clip(processed_real_image, 0, 1)*255).astype(np.uint8) # convert to image format
    #
    #     # convert from numpy image to torch tensor
    #     processed_real_image_pt = self.to_tensor(processed_real_image)
    #
    #     # generate an image
    #     gen_sim_image = self.generator(processed_real_image_pt)
    #
    #     # convert to numpy, image format, size expected by rl agent
    #     gen_sim_image = gen_sim_image[0,0,...].detach().cpu().numpy() # pytorch batch -> numpy image
    #     gen_sim_image = (np.clip(gen_sim_image, 0, 1)*255).astype(np.uint8) # convert to image format
    #
    #     if self.params['dims'] != self.rl_image_size:
    #         gen_sim_image = cv2.resize(gen_sim_image, tuple(self.rl_image_size), interpolation=cv2.INTER_NEAREST) # resize to RL expected
    #
    #     return gen_sim_image, processed_real_image_plot
