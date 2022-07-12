import unittest
import os
import torch

# # hacky way of importing from repo
import sys; sys.path.append('..'); sys.path.append('.')
# from os.path import split, realpath, join
# base_dir = split(split(realpath(__file__))[0])[0]
# sys.path.append(base_dir)

from trainers.data_loader import image_handler as image_loader
from trainers.train_single_task import trainer
from gan_models.models_128 import GeneratorUNet, weights_init_normal

# tests
# ====================================================


class test_data(unittest.TestCase):
    def setUp(self):
        # put any class contructing code in here and will be called before any tests
        self.dir = 'pytests'
        self.size = 128
        self.dataset_train = image_loader(base_dir=self.dir, size=self.size)
        self.dataset_val = image_loader(base_dir=self.dir, size=self.size, val=True)
        self.generator = GeneratorUNet(in_channels=1, out_channels=1)
        self.generator.apply(weights_init_normal)
        self.train = trainer(self.dataset_train,
                             self.dataset_val,
                             self.generator,
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
