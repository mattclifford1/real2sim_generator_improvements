'''
Script to test UNet against the validation set and a given discriminator.
Code adapted from Alex Church's tactile_gym_sim2real repo

Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk
'''
import torch
import os
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from utils.image_transforms import GAN_io
from gan_models.models_128 import GeneratorUNet, Discriminator


def show_ims(ims):
    fig, axs = plt.subplots(len(ims))
    for i in range(len(ims)):
        axs[i].imshow(ims[i])
    plt.show()

def model_to_device(model):
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        model = model.cuda()
    else:
        model = model
    return model

class gan_tester():
    def __init__(self, gen_model_dir, discrim_model_dir, image_size=[128, 128]):
        self.gen_model_dir = gen_model_dir
        self.discrim_model_dir = discrim_model_dir
        self.image_size = image_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_generator()
        self.init_discrim()

    def init_generator(self):
        self.generator = GeneratorUNet(in_channels=1, out_channels=1)
        self.generator.load_state_dict(torch.load(os.path.join(self.gen_model_dir, 'checkpoints/best_generator.pth'), map_location=torch.device(self.device)))
        self.generator = model_to_device(self.generator)
        self.generator.eval()
        # get image io helper
        self.gen_ims_io = GAN_io(self.gen_model_dir, rl_image_size=self.image_size)

    def init_discrim(self):
        self.discriminator = Discriminator(in_channels=1)
        self.discriminator.load_state_dict(torch.load(os.path.join(self.discrim_model_dir, 'checkpoints/best_discriminator.pth'), map_location=torch.device(self.device)))
        self.discriminator = model_to_device(self.discriminator)
        self.discriminator.eval()

    def load_ims(self, real_image_file, sim_image_file):
        image_real = io.imread(real_image_file)
        image_sim = io.imread(sim_image_file)
        # preprocess image
        im_preprocced = self.gen_ims_io.process_raw_image(image_real)
        # convert to tensors
        im_real_pt = self.gen_ims_io.to_tensor(im_preprocced)
        im_sim_pt = self.gen_ims_io.to_tensor(image_sim)
        return im_real_pt, im_sim_pt, image_sim

    def get_info(self, image_real, image_sim):
        im_real_pt, im_sim_pt, image_sim = self.load_ims(image_real, image_sim)
        pred_sim = self.generator(im_real_pt)
        metrics_dict = get_metrics(pred_sim, im_real_pt, image_sim, self.gen_ims_io)
        # discriminator is conditions the generated/simulated image with the real camera image
        # img_input = torch.cat((pred_sim, im_real_pt), 1)
        discrim_out = self.discriminator(pred_sim, im_real_pt)
        discrim_avg_score = discrim_out.detach().cpu().numpy().mean()
        metrics_dict['Score on discriminator (accuracy)'] = discrim_avg_score
        return metrics_dict

    def get_all_model_weights(self):
        weights = {}
        for name, param in self.generator.named_parameters():
            weights[name] = param
        return weights


def get_metrics(pred_sim, im_real_pt, image_sim, ims_io):
    gen_sim_npy = ims_io.to_numpy(pred_sim)
    # show_ims([image, processed_real_image, gen_sim_npy, image_sim])
    # MSE = (gen_sim_npy - image_sim).mean()
    gen_im_flt = img_as_float(gen_sim_npy)
    sim_im_flt = img_as_float(image_sim)
    mse = mean_squared_error(sim_im_flt, gen_im_flt)
    # ssim = SSIM(sim_im_flt, gen_im_flt, data_range=gen_im_flt.max() - gen_im_flt.min())
    # ssim, diff_image = SSIM(sim_im_flt, gen_im_flt, full=True)
    ssim = SSIM(sim_im_flt, gen_im_flt)
    return {'MSE':mse, 'SSIM':ssim}

def get_all_test_ims(dir, ext='.png'):
    ims = []
    for im in os.listdir(dir):
        if os.path.splitext(im)[1] == ext:
            ims.append(im)
    return ims

def get_weights(gen_model_dir, discrim_model_dir):
    tester = gan_tester(gen_model_dir, discrim_model_dir)
    weights = tester.get_all_model_weights()
    layers = []
    for name in weights.keys():
        print(name, ': ', weights[name].mean())
        layers.append(weights[name])
    return weights, layers


def run(gen_model_dir, discrim_model_dir, real_images_dir, sim_images_dir, dev=False):
    tester = gan_tester(gen_model_dir, discrim_model_dir)

    all_metrics = {}
    MSEs = []
    discrim_scores = []
    i = 0
    for image in tqdm(get_all_test_ims(real_images_dir), desc="All images", leave=False):
        # get image pair
        test_image_real = os.path.join(real_images_dir, image)
        test_image_sim = os.path.join(sim_images_dir, image)
        # now test image pair
        metrics_dict = tester.get_info(test_image_real, test_image_sim)
        if i == 0:
            # init keys
            for key in metrics_dict.keys():
                all_metrics[key] = []
        for key in metrics_dict.keys():
            all_metrics[key].append(metrics_dict[key])
        if i == 2 and dev == True:
            break
        i+=1

    return all_metrics


if __name__ == '__main__':
    '''
    input:
    - test dataset
    - generator
    - discriminator
    output:
    - test score (MSE)
    - discriminator score (error etc.)
    '''
    gan_model_dir = 'no_git_data/128x128_tap_250epochs/'
    gan_model_dir = '../models/sim2real/alex/trained_gans/[edge_2d]/128x128_[tap]_250epochs'
    # gan_model_dir = '../models/sim2real/alex/trained_gans/[edge_2d]/128x128_[shear]_250epochs'
    # gan_model_dir = '../models/sim2real/alex/trained_gans/[surface_3d]/128x128_[shear]_250epochs'


    real_images_dir = '../data/Bourne/tactip/real/edge_2d/tap/csv_val/images'
    sim_images_dir = '../data/Bourne/tactip/sim/edge_2d/tap/128x128/csv_val/images'

    MSEs, discrim_scores = run(gan_model_dir, real_images_dir, sim_images_dir, dev=True)
    print(np.mean(MSEs))
    print(np.mean(discrim_scores))
