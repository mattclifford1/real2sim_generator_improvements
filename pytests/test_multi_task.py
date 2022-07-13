import unittest
import os
import torch

# # hacky way of importing from repo
import sys; sys.path.append('..'); sys.path.append('.')
# from os.path import split, realpath, join
# base_dir = split(split(realpath(__file__))[0])[0]
# sys.path.append(base_dir)

from trainers.data_loader import image_handler as image_loader
from trainers.train_multi_task import trainer, get_all_loaders
from gan_models.models_128 import GeneratorUNet, Discriminator, weights_init_normal

# tests
# ====================================================


class test_data(unittest.TestCase):
    def setUp(self):
        # put any class contructing code in here and will be called before any tests
        self.dir = 'pytests'
        self.size = 128
        self.datasets_train, self.datasets_val = get_all_loaders(self.dir)
        self.generator = GeneratorUNet(in_channels=1, out_channels=1)
        self.discriminator = Discriminator(in_channels=1)
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        self.train = trainer(self.datasets_train,
                             self.datasets_val,
                             self.generator,
                             self.discriminator,
                             save_dir=os.path.join(self.dir, 'models', 'sim2real', 'matt'),
                             batch_size=2,
                             epochs=1)

    def test_run_epoch(self):
        # train from scratch
        self.train.start()
        # now train new run
        self.train.start()



if __name__ == '__main__':
    unittest.main()
