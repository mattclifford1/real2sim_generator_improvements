import unittest
import os
import torch

# # hacky way of importing from repo
import sys; sys.path.append('..'); sys.path.append('.')
# from os.path import split, realpath, join
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
        self.h = l.image_handler(base_dir=self.dir, size=self.size)
        self.h_ram = l.image_handler(base_dir=self.dir, size=self.size, store_ram=True)

    def test_can_make_loader(self):
        assert isinstance(self.h, l.image_handler)
        assert isinstance(self.h_ram, l.image_handler)

    def test_can_get_sample(self):
        sample = self.h[0]
        assert isinstance(sample['real'], torch.Tensor)
        assert isinstance(sample['sim'], torch.Tensor)
        sample = self.h_ram[0]
        assert isinstance(sample['real'], torch.Tensor)
        assert isinstance(sample['sim'], torch.Tensor)

    def test_sample_dims(self):
        sample = self.h[0]
        assert sample['real'].shape == (1, self.size, self.size)
        assert sample['sim'].shape == (1, self.size, self.size)
        sample = self.h_ram[0]
        assert sample['real'].shape == (1, self.size, self.size)
        assert sample['sim'].shape == (1, self.size, self.size)

    def test_accepts_torch_loader(self):
        batch_size=2
        loader = torch.utils.data.DataLoader(self.h,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=batch_size)
        for i, batch in enumerate(loader):
            if i > 1:
                break
        assert batch is not None
        assert batch['real'].shape == (batch_size, 1, self.size, self.size)

    def test_accepts_torch_loader_ram(self):
        batch_size=2
        loader = torch.utils.data.DataLoader(self.h_ram,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=batch_size)
        for i, batch in enumerate(loader):
            if i > 1:
                break
        assert batch is not None
        assert batch['real'].shape == (batch_size, 1, self.size, self.size)

    def test_bad_task_input_error(self):
        with self.assertRaises(Exception):
            x = l.image_handler(base_dir=self.dir, size=self.size, task=['1', '2'])
        with self.assertRaises(Exception):
            x = l.image_handler(base_dir=self.dir, size=self.size, task=['surface_3d', '2'])
        with self.assertRaises(Exception):
            x = l.image_handler(base_dir=self.dir, size=self.size, task=['surface_3d'])
        with self.assertRaises(Exception):
            x = l.image_handler(base_dir=self.dir, size=self.size, task=['surface_3d', 'tap', '1'])
        with self.assertRaises(Exception):
            x = l.image_handler(base_dir=self.dir, size=self.size, task=[])

    def test_maybe_lower_data_size(self):
        # extra test if the full data set is present - test if we can lower the data size
        if os.path.isdir(os.path.join('..', 'data/Bourne/tactip', 'real', 'edge_2d', 'tap', 'csv_train')):
            full_data = l.image_handler(base_dir='..', size=self.size)
            assert len(full_data) == 5000
            half_data = l.image_handler(base_dir='..', size=self.size, use_percentage_of_data=0.5)
            assert len(half_data) == 2500
            q_data = l.image_handler(base_dir='..', size=self.size, use_percentage_of_data=0.25)
            assert len(q_data) == 1250
            with self.assertRaises(Exception):
                x = l.image_handler(base_dir='..', size=self.size, use_percentage_of_data=-1)
            with self.assertRaises(Exception):
                x = l.image_handler(base_dir='..', size=self.size, use_percentage_of_data=1.1)



if __name__ == '__main__':
    unittest.main()
