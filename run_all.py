import os
from argparse import ArgumentParser
from gan_tester import run
import numpy as np

if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--dir", default='..', help='path to folder where data and models are held')
    ARGS = parser.parse_args()

    gan_model_dir = os.path.join(ARGS.dir, 'models/sim2real/alex/trained_gans/[edge_2d]/128x128_[tap]_250epochs')
    real_images_dir = os.path.join(ARGS.dir, 'data/Bourne/tactip/real/edge_2d/tap/csv_val/images')
    sim_images_dir = os.path.join(ARGS.dir, 'data/Bourne/tactip/sim/edge_2d/tap/128x128/csv_val/images')
    MSEs, discrim_scores = run(gan_model_dir, real_images_dir, sim_images_dir)
    print(np.mean(MSEs))
    print(np.mean(discrim_scores))
