import unittest
import os
import torch

from os.path import split, realpath, join
# # hacky way of importing from repo
import sys; sys.path.append('..'); sys.path.append('.')
# base_dir = split(split(realpath(__file__))[0])[0]
# sys.path.append(base_dir)

from run_tests.diff_data_for_generators import get_results

# tests
# ====================================================


class test_data(unittest.TestCase):
    def setUp(self):
        # put any class contructing code in here and will be called before any tests
        self.dir = '..'

    def test_get_metrics(self):
        metrics = get_results(dir=self.dir, gen=('edge_2d','tap'), data=('edge_2d', 'tap'), dev=True)
        for key in metrics.keys():
            assert type(metrics[key]) == list



if __name__ == '__main__':
    unittest.main()
