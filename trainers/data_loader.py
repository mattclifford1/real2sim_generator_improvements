'''
Load sim and real data with torch data loader

Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk
'''
import os
from argparse import ArgumentParser
import sys; sys.path.append('..'); sys.path.append('.')

def get_all_test_ims(dir, ext='.png'):
    ims = []
    for im in os.listdir(dir):
        if os.path.splitext(im)[1] == ext:
            ims.append(im)
    return ims



if __name__ == '__main__':
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    ARGS = parser.parse_args()

    data = ('edge_2d', 'tap')
    real_images_dir = os.path.join(ARGS.dir, 'data/Bourne/tactip/real/'+data[0]+'/'+data[1]+'/csv_val/images')
    sim_images_dir = os.path.join(ARGS.dir, 'data/Bourne/tactip/sim/'+data[0]+'/'+data[1]+'/128x128/csv_val/images')
