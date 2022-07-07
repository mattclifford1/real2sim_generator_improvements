import unittest
import os
import torch

from os.path import split, realpath, join
# # hacky way of importing from repo
import sys; sys.path.append('..'); sys.path.append('.')
# base_dir = split(split(realpath(__file__))[0])[0]
# sys.path.append(base_dir)

import trainers.data_loader as l

# tests
# ====================================================


class test_data(unittest.TestCase):
    def setUp(self):
        # put any class contructing code in here and will be called before any tests
        self.dir = 'pytests'
        self.size = 128
        self.h = l.image_handler(size=self.size)

    def test_can_make_loader(self):
        assert isinstance(self.h, l.image_handler)

    def test_can_get_sample(self):
        sample = self.h[0]
        assert isinstance(sample['real'], torch.Tensor)
        assert isinstance(sample['sim'], torch.Tensor)

    def test_sample_dims(self):
        sample = self.h[0]
        assert sample['real'].shape == (1, self.size, self.size)
        assert sample['sim'].shape == (1, self.size, self.size)


if __name__ == '__main__':
    unittest.main()
