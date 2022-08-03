'''utils to handle any pytorch related task eg. generator or pose estimation'''
import torch
import os
import numpy as np
import sys; sys.path.append('..'); sys.path.append('.')
from gan_models.models_128 import GeneratorUNet, weights_init_normal, weights_init_pretrained


def preprocess_numpy_image(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)   # make into batch one 1
    image = torch.from_numpy(image)
    return image

def post_process_torch_image(image):
    image = image.cpu().detach().numpy()
    image = image[0, :, :, :]
    image = image.transpose((1, 2, 0))
    return np.clip(image, 0, 1)

'''generator model loader and get prediction of a single image'''
class generator():
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = GeneratorUNet(in_channels=1, out_channels=1)
        if os.path.isfile(weights_path):
            weights_init_pretrained(self.generator, weights_path)
        else:
            print('Could not find weights path: '+str(weights_path)+'\nInitialising generator with random weights')
            self.generator.apply(weights_init_normal)

    def get_prediction(self, image):
        '''preprocess image'''
        image = preprocess_numpy_image(image)
        image.to(device=self.device, dtype=torch.float)
        ''' get prediction'''
        pred = self.generator(image)
        return post_process_torch_image(pred)


'''pose estimation model on single image'''
class pose_estimation():
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.isfile(weights_path):
            print('*******\n*******\nCannot load pose estimation weights path: '+str(weights_path))

    def get_prediction(self, image):
        '''preprocess image'''
        iamge = preprocess_numpy_image(image)
        image.to(device=self.device, dtype=torch.float)
        ''' get prediction'''
        return 0  # implement
