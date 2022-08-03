import sys
import argparse
import threading
import time
import pickle
import os
import pandas as pd

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QInputDialog, QLineEdit, QMenu, QFileDialog, QPushButton, QGridLayout, QLabel, QSlider, QComboBox, QCheckBox
# from PyQt6.QtGui import QImage, QColor
from PyQt6.QtCore import Qt

from PyQt6.QtGui import QPainter, QBrush
from PyQt6.QtWidgets import QStyle, QStyleOptionSlider
from PyQt6.QtCore import QRect, QPoint, Qt

import sys; sys.path.append('..'); sys.path.append('.')
from gui_utils import change_im, load_image
import gui_utils
import net_utils

class make_app(QMainWindow):
    def __init__(self, app, args):
        super().__init__()
        self.args = args
        self.app = app
        # self.set_window()
        self.copy_or_real = 'Copy'
        self.generator = net_utils.generator(self.args.generator_path)
        self.init_images()
        self.init_widgets()
        self.init_layout()
        self.image_display_size = (256, 256)
        self.display_images()

    def set_window(self):
        '''
        set the application window to the size of the screen
        '''
        screen = self.app.primaryScreen()
        size = screen.size()
        self.width = size.width() - 50
        self.height =  size.height() - 50

    def init_images(self, screen=False):
        '''
        make blank images to place on screen before actual image is chosen
        this creates the UI to be the correct size
        '''
        # make image placeholders
        pad = 30
        pad_small = 10
        height_reduction_factor = 1.6
        self.width_ratio = 1 # 16/9
        if screen:
            self.height = int(self.height/height_reduction_factor) - pad
        else:
            self.height = int(256)
        self.width = int(self.height*self.width_ratio)

        # set up ims and click functions
        self.im_Qlabels = {'im_reference':QLabel(self),
        'im_compare':QLabel(self),
        # 'im_diff':QLabel(self)
        }
        self.im_Qlabels['im_reference'].setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.im_Qlabels['im_compare'].setAlignment(Qt.AlignmentFlag.AlignRight)

        # load images
        self.sensor_data = {}
        if os.path.exists(self.args.image_path):
            self.sensor_data['im_compare'] = load_image(self.args.image_path)
        else:
            self.sensor_data['im_compare'] = np.zeros([128, 128, 1], dtype=np.uint8)
        if os.path.exists(self.args.csv_path):
            self.load_dataset(self.args.csv_path)
        else:
            self.sensor_data['im_reference'] = np.zeros([128, 128, 1], dtype=np.uint8)

        # self.im_Qlabels['colour_output'].mousePressEvent = self.image_click
        # hold video frames
        # dummy_frame = [self.dummy_im]*2
        # self.input_frames = {'UI_colour_original':dummy_frame,
                             # 'UI_depth':dummy_frame}
        # self.num_frames = len(self.input_frames['UI_colour_original'])

    def init_widgets(self):
        '''
        create all the widgets we need and init params
        '''
        '''buttons'''
        self.button_dataset_load = QPushButton('Choose Dataset', self)
        self.button_dataset_load.clicked.connect(self.choose_dataset)
        self.button_prev = QPushButton('<', self)
        self.button_prev.clicked.connect(self.load_prev_image)
        self.button_next = QPushButton('>', self)
        self.button_next.clicked.connect(self.load_next_image)
        # self.button_image2 = QPushButton('Choose Image 2', self)
        # self.button_image2.clicked.connect(self.choose_image_compare)
        '''checkboxes'''
        self.checkbox_copy_im = QCheckBox('Real Image', self)
        self.checkbox_copy_im.toggled.connect(self.copy_ref_im)
        self.run_generator = False
        self.checkbox_generator = QCheckBox('Run Generator', self)
        self.checkbox_generator.toggled.connect(self.toggle_generator)
        self.checkbox_generator.setEnabled(False)
        '''sliders'''
        # rotation
        self.im_trans_params = {}
        self.slider_rotation = QSlider(Qt.Orientation.Horizontal)
        self.slider_rotation.setMinimum(-180)
        self.slider_rotation.setMaximum(180)
        self.slider_rotation.valueChanged.connect(self.rotate_image)
        self.slider_rotation.valueChanged.connect(self.display_images)
        self.slider_rotation.sliderReleased.connect(self.display_images)
        self.im_trans_params['angle_to_rotate'] = 0
        self.slider_rotation.setValue(self.im_trans_params['angle_to_rotate'])
        self.label_rotation = QLabel(self)
        self.label_rotation.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.label_rotation.setText('Rotation:')
        self.label_rotation_value = QLabel(self)
        self.label_rotation_value.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.label_rotation_value.setText(str(self.im_trans_params['angle_to_rotate']))
        # guassian blur
        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setMinimum(0)
        self.slider_blur.setMaximum(40)
        self.slider_blur.valueChanged.connect(self.blur_image)
        self.slider_blur.sliderReleased.connect(self.display_images)
        self.im_trans_params['guass_blur_kern_size'] = 0
        self.slider_blur.setValue(self.im_trans_params['guass_blur_kern_size'])
        self.label_blur = QLabel(self)
        self.label_blur.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.label_blur.setText('Blur:')
        self.label_blur_value = QLabel(self)
        self.label_blur_value.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.label_blur_value.setText(str(self.im_trans_params['guass_blur_kern_size']))
        # brightness
        self.slider_brightness = QSlider(Qt.Orientation.Horizontal)
        self.slider_brightness.setMinimum(-255)
        self.slider_brightness.setMaximum(255)
        self.slider_brightness.valueChanged.connect(self.brightness_image)
        self.slider_brightness.valueChanged.connect(self.display_images)
        self.slider_brightness.sliderReleased.connect(self.display_images)
        self.im_trans_params['brightness_adjustment'] = 0
        self.slider_brightness.setValue(self.im_trans_params['brightness_adjustment'])
        self.label_brightness = QLabel(self)
        self.label_brightness.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.label_brightness.setText('Brightness:')
        self.label_brightness_value = QLabel(self)
        self.label_brightness_value.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.label_brightness_value.setText(str(self.im_trans_params['brightness_adjustment']))

    def init_layout(self):
        '''
        place all the widgets in the window
        '''
        # make main widget insdie the QMainWindow
        self.main_widget = QWidget()
        self.layout = QGridLayout()
        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)
        # sizes
        im_width = 20
        im_height = 20
        button = 1
        slider_width = 10
        check_box_width = 5
        # horizonal start values
        start_im = 0
        start_controls = im_width*2

        # display images
        self.layout.addWidget(self.im_Qlabels['im_reference'], start_im, start_im,   im_height, im_width)
        self.layout.addWidget(self.im_Qlabels['im_compare'],   start_im, im_width,   im_height, im_width)
        # self.layout.addWidget(self.im_Qlabels['im_diff'], im_height+1, im_width//2, im_height//2, im_width//2)

        # load files
        # self.layout.addWidget(self.button_dataset_load, im_height, 1, 1, 1)

        # image buttons (prev, copy, next, etc.)
        self.layout.addWidget(self.button_prev,    im_height, 0, 1, 1)
        # self.layout.addWidget(self.button_copy_im, im_height, 1, 1, 1)
        self.layout.addWidget(self.button_next,    im_height, 2, 1, 1)

        # checkboxes
        self.layout.addWidget(self.checkbox_copy_im,   button*2, start_controls+button, button, check_box_width)
        self.layout.addWidget(self.checkbox_generator, button*2, start_controls+button+check_box_width, button, check_box_width)

        # sliders
        self.layout.addWidget(self.slider_rotation,   button*3, start_controls+button, button, slider_width)
        self.layout.addWidget(self.label_rotation,    button*3, start_controls,   button, button)
        self.layout.addWidget(self.label_rotation_value,    button*3, start_controls+button+slider_width,   button, button)

        self.layout.addWidget(self.slider_blur,       button*4, start_controls+button, button, slider_width)
        self.layout.addWidget(self.label_blur,        button*4, start_controls,   button, button)
        self.layout.addWidget(self.label_blur_value,    button*4, start_controls+button+slider_width,   button, button)

        self.layout.addWidget(self.slider_brightness, button*5, start_controls+button, button, slider_width)
        self.layout.addWidget(self.label_brightness,  button*5, start_controls,   button, button)
        self.layout.addWidget(self.label_brightness_value,    button*5, start_controls+button+slider_width,   button, button)
        # init it!
        self.show()

    '''
    ==================== functions to bind to widgets ====================
    '''
    def load_dataset(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.im_num = 0   # row of csv dataset to use
        self.im_sim_dir = os.path.join(os.path.dirname(csv_file), 'images')
        self.im_real_dir = os.path.join(os.path.dirname(gui_utils.get_real_csv_given_sim(csv_file)), 'images')
        self.load_sim_image()
        self.load_real_image()

    def load_sim_image(self):
        image = self.df.iloc[self.im_num]['sensor_image']
        image_path = os.path.join(self.im_sim_dir, image)
        poses = ['pose_'+str(i) for i in range(1,7)]
        self.sensor_data['poses'] = {}
        for pose in poses:
            self.sensor_data['poses'][pose] = self.df.iloc[self.im_num][pose]
        self.sensor_data['im_reference'] = load_image(image_path)
        self.sensor_data['im_reference'] = gui_utils.process_im(self.sensor_data['im_reference'], data_type='sim')

    def load_real_image(self):
        if self.copy_or_real == 'Real':
            image = self.df.iloc[self.im_num]['sensor_image']
            image_path = os.path.join(self.im_real_dir, image)
            self.sensor_data['im_compare'] = load_image(image_path)
            self.sensor_data['im_compare'] = gui_utils.process_im(self.sensor_data['im_compare'], data_type='real')
            if self.run_generator == True:
                self.sensor_data['im_compare'] = self.generator.get_prediction(self.sensor_data['im_compare'])
        elif self.copy_or_real == 'Copy':
            self.sensor_data['im_compare'] = self.sensor_data['im_reference'].copy()


    def load_prev_image(self):
        self.im_num -= 1
        if self.im_num < 0:
            self.im_num = len(self.df) - 1
        self.load_sim_image()
        self.load_real_image()
        self.display_images()

    def load_next_image(self):
        self.im_num += 1
        if self.im_num == len(self.df):
            self.im_num = 0
        self.load_sim_image()
        self.load_real_image()
        self.display_images()

    def copy_ref_im(self):
        if self.copy_or_real == 'Real':
            self.copy_or_real = 'Copy'
            self.checkbox_generator.setEnabled(False)
        elif self.copy_or_real == 'Copy':
            self.copy_or_real = 'Real'
            self.checkbox_generator.setEnabled(True)
        self.load_real_image()
        self.display_images()

    def toggle_generator(self):
        self.run_generator = not self.run_generator
        self.load_real_image()
        self.display_images()

    def choose_dataset(self):
        '''
        choose video file from Files
        '''
        try:
            csv_file, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;CSV files (*.csv)")#, options=options)
            if csv_file != '':
                self.load_dataset(csv_file)
                self.display_images()
                # self.reset_all_sliders()
        except:
            self.statusBar().showMessage('Cancelled Load')

    def choose_image_compare(self):
        '''
        choose video file from Files
        '''
        try:
            self.image2_file, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Image Files (*.png)")#, options=options)
            if self.image2_file != '':
                # self.reset_all_sliders()
                # self.get_list_of_ims(self.video_file, button=self.button_video)
                print('now impliment load image')
        except:
            self.statusBar().showMessage('Cancelled Load')

    def rotate_image(self):
        self.im_trans_params['angle_to_rotate'] = self.slider_rotation.value()
        self.label_rotation_value.setText(str(self.im_trans_params['angle_to_rotate']))

    def blur_image(self):
        self.im_trans_params['guass_blur_kern_size'] = (int(self.slider_blur.value()/2)*2) + 1    # need to make kernel size odd
        self.label_blur_value.setText(str(self.im_trans_params['guass_blur_kern_size']))

    def brightness_image(self):
        self.im_trans_params['brightness_adjustment'] = self.slider_brightness.value()
        self.label_brightness_value.setText(str(self.im_trans_params['brightness_adjustment']))
        self.im_trans_params['brightness_adjustment'] = self.im_trans_params['brightness_adjustment']/255

    '''
    image updaters
    '''
    def transform_compare_image(self):

        return gui_utils.transform_image(self.sensor_data['im_compare'], self.im_trans_params)

    def display_images(self):
        change_im(self.im_Qlabels['im_reference'], self.sensor_data['im_reference'], resize=self.image_display_size)
        change_im(self.im_Qlabels['im_compare'], self.transform_compare_image(), resize=self.image_display_size)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='image file to use', default=os.path.join(os.path.expanduser('~'),'summer-project/data/Bourne/tactip/sim/surface_3d/tap/128x128/csv_train/images/image_1.png'))
    parser.add_argument('--csv_path', type=str, help='csv file to use', default=os.path.join(os.path.expanduser('~'),'summer-project/data/Bourne/tactip/sim/surface_3d/shear/128x128/csv_train/targets.csv'))
    parser.add_argument('--generator_path', type=str, help='generator weights file to use', default=os.path.join(os.path.expanduser('~'),'summer-project/models/sim2real/matt/surface_3d/shear/pretrained_edge_tap/no_ganLR:0.0002_decay:0.1_BS:64_DS:1.0/run_0/models/best_generator.pth'))
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = make_app(app, args)
    sys.exit(app.exec())
