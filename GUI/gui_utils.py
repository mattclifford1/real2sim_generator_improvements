from skimage.transform import resize, rotate
from skimage.util import img_as_ubyte
import cv2
import os
import numpy as np
from PyQt6.QtGui import QPixmap, QImage
# from PyQt5.QtWidgets import qApp
from PyQt6.QtWidgets import QApplication

'''
image helper functions
'''
def resize_im_to(np_array, size):
    down_im = resize(np_array, size)
    return img_as_ubyte(down_im)

def change_im(widget, im, resize=False, return_qimage=False):
    '''
    given a numpy image, changes the given widget Frame
    '''
    if im.shape[2] == 1:
        im = np.concatenate([im, im, im], axis=2)
    if resize:
        im = resize_im_to(im, resize)
    qimage = QImage(im,
                    im.shape[1],
                    im.shape[0],
                    im.shape[1]*im.shape[2],
                    QImage.Format.Format_RGB888)
                    # QImage.Format_RGB888)  # PyQt5
    pixmap = QPixmap(qimage)
    widget.setPixmap(pixmap)
    # qApp.processEvents()   # force to change other UI wont respond
    QApplication.processEvents()   # force to change other UI wont respond
    if return_qimage:
        return qimage

def load_image(image_path):
    return cv2.imread(image_path)

def load_and_crop_raw_real(image_path):
    image = cv2.imread(image_path)
    # Convert to gray scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Add channel axis
    image = image[..., np.newaxis]
    # Crop to specified bounding box
    bbox = [80,25,530,475]
    x0, y0, x1, y1 = bbox
    image = image[y0:y1, x0:x1]
    # Resize to specified dims
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    # Add channel axis
    image = image[..., np.newaxis]
    return image.astype(np.float32) / 255.0

def process_im(image, data_type='sim'):
    if data_type == 'real':
        image = image*255
        image = image.astype(np.uint8)
        # threshold_image
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -30)
        image = image[..., np.newaxis]
    elif data_type == 'sim':
        # Convert to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Add channel axis
        image = image[..., np.newaxis]
    return image.astype(np.float32) / 255.0

def get_real_csv_given_sim(sim_csv):
    ''' try find equivelant real csv given a sim
        only works for standard dir structing
        will return sim_csv if can't find real csv'''
    dirs = sim_csv.split(os.sep)
    dirs[0] = os.sep
    dirs.pop(-3)       # remove 128x128
    dirs[-5] = 'real'  # swap sim for real
    real_csv = os.path.join(*dirs)
    if os.path.isfile(real_csv):
        return real_csv
    else:
        return sim_csv

def transform_image(image, params_dict):
    if 'angle_to_rotate' in params_dict.keys():
        image = rotate(image, params_dict['angle_to_rotate'])

    if 'brightness_adjustment' in params_dict.keys():
        image = np.clip(image + params_dict['brightness_adjustment'], 0, 1)

    if 'guass_blur_kern_size' in params_dict.keys():
        if params_dict['guass_blur_kern_size'] > 0:
            image = cv2.GaussianBlur(image,(params_dict['guass_blur_kern_size'], params_dict['guass_blur_kern_size']), cv2.BORDER_DEFAULT)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
    return image


if __name__ == '__main__':
    get_real_csv_given_sim('/home/matt/summer-project/data/Bourne/tactip/sim/edge_2d/shear/128x128/csv_train/targets.csv')
