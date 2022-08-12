'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

Print number of paramters in each model
'''
import torch
import sys; sys.path.append('..'); sys.path.append('.')
import downstream_task.networks.model_128 as pose_128
from gan_models.models_128 import GeneratorUNet, Discriminator


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    pose = net = pose_128.network(final_size=3)
    generator = GeneratorUNet(in_channels=1, out_channels=1)
    discriminator = Discriminator(in_channels=1)

    print('PoseNet:       ', count_parameters(pose))
    print('Generator:     ', count_parameters(generator))
    print('Discriminator: ', count_parameters(discriminator))
