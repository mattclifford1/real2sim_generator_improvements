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
import metrics_utils

class make_app(QMainWindow):
    def __init__(self, app, args):
        super().__init__()
        self.args = args
        self.app = app
        # self.set_window()
        self.copy_or_real = 'Copy'
        self.generator = net_utils.generator(self.args.generator_path)
        self.pose_esimator_sim = net_utils.pose_estimation(self.args.pose_path, sim=True)
        self.pose_esimator_real = net_utils.pose_estimation(self.args.pose_path, sim=False)
        self.metrics = metrics_utils.im_metrics()
        self.init_widgets()
        self.init_images()
        self.init_layout()
        self.image_display_size = (200, 200)
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

        # load images
        self.sensor_data = {}
        # if os.path.exists(self.args.image_path):
        #     self.sensor_data['im_compare'] = load_image(self.args.image_path)
        # else:
        #     self.sensor_data['im_compare'] = np.zeros([128, 128, 1], dtype=np.uint8)
        if os.path.exists(self.args.csv_path):
            self.load_dataset(self.args.csv_path)
        else:
            self.sensor_data['im_sim'] = np.zeros([128, 128, 1], dtype=np.uint8)

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
        # set up ims
        self.im_Qlabels = {'im_real':QLabel(self),
                           'im_real_trans':QLabel(self),
                           'im_gen':QLabel(self),
                           'im_gen_trans':QLabel(self),
                           'im_sim':QLabel(self),
                           'im_sim_trans':QLabel(self),
                           }
        for key in self.im_Qlabels.keys():
            self.im_Qlabels[key].setAlignment(Qt.AlignmentFlag.AlignCenter)
        # UI widgets
        self.widgets = {'button': {}, 'slider': {}, 'checkbox': {}, 'label': {}}
        '''im labels'''
        self.widgets['label']['im_real'] = QLabel(self)
        self.widgets['label']['im_real'].setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.widgets['label']['im_real'].setText('Real Image')
        self.widgets['label']['im_real_trans'] = QLabel(self)
        self.widgets['label']['im_real_trans'].setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.widgets['label']['im_real_trans'].setText('Real Image Trans')
        self.widgets['label']['im_gen'] = QLabel(self)
        self.widgets['label']['im_gen'].setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.widgets['label']['im_gen'].setText('Generatated Image')
        self.widgets['label']['im_gen_trans'] = QLabel(self)
        self.widgets['label']['im_gen_trans'].setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.widgets['label']['im_gen_trans'].setText('Generatated Image Trans')
        self.widgets['label']['im_sim'] = QLabel(self)
        self.widgets['label']['im_sim'].setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.widgets['label']['im_sim'].setText('Simulated Image')
        self.widgets['label']['im_sim_trans'] = QLabel(self)
        self.widgets['label']['im_sim_trans'].setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.widgets['label']['im_sim_trans'].setText('Simulated Image Trans')
        '''buttons'''
        # self.widgets['button']['load_dataset'] = QPushButton('Choose Dataset', self)
        # self.widgets['button']['load_dataset'].clicked.connect(self.choose_dataset)
        self.widgets['button']['prev'] = QPushButton('<', self)
        self.widgets['button']['prev'].clicked.connect(self.load_prev_image)
        self.widgets['label']['filename'] = QLabel(self)
        self.widgets['label']['filename'].setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.widgets['label']['filename'].setText('')
        self.widgets['button']['next'] = QPushButton('>', self)
        self.widgets['button']['next'].clicked.connect(self.load_next_image)
        self.widgets['button']['reset_sliders'] = QPushButton('Reset', self)
        self.widgets['button']['reset_sliders'].clicked.connect(self.reset_sliders)
        self.widgets['button']['force_update'] = QPushButton('Update', self)
        self.widgets['button']['force_update'].clicked.connect(self.display_images)
        # self.button_image2 = QPushButton('Choose Image 2', self)
        # self.button_image2.clicked.connect(self.choose_image_compare)
        '''checkboxes'''
        # self.widgets['checkbox']['real_im'] = QCheckBox('Real Image', self)
        # self.widgets['checkbox']['real_im'].toggled.connect(self.copy_ref_im)
        # self.run_generator = False
        # self.widgets['checkbox']['run_generator'] = QCheckBox('Run Generator', self)
        # self.widgets['checkbox']['run_generator'].toggled.connect(self.toggle_generator)
        # self.widgets['checkbox']['run_generator'].setEnabled(False)
        '''sliders'''
        # rotation
        self.im_trans_params = {}
        self.widgets['slider']['rotation'] = QSlider(Qt.Orientation.Horizontal)
        self.widgets['slider']['rotation'].setMinimum(-180)
        self.widgets['slider']['rotation'].setMaximum(180)
        self.widgets['slider']['rotation'].valueChanged.connect(self.rotate_value_change)
        self.widgets['slider']['rotation'].valueChanged.connect(self._display_images_quick)
        self.widgets['slider']['rotation'].sliderReleased.connect(self.display_images)
        self.im_trans_params['angle_to_rotate'] = 0
        self.widgets['slider']['rotation'].setValue(self.im_trans_params['angle_to_rotate'])
        self.widgets['label']['rotation'] = QLabel(self)
        self.widgets['label']['rotation'].setAlignment(Qt.AlignmentFlag.AlignRight)
        self.widgets['label']['rotation'].setText('Rotation:')
        self.widgets['label']['rotation_value'] = QLabel(self)
        self.widgets['label']['rotation_value'].setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.widgets['label']['rotation_value'].setText(str(self.im_trans_params['angle_to_rotate']))
        # guassian blur
        self.widgets['slider']['blur'] = QSlider(Qt.Orientation.Horizontal)
        self.widgets['slider']['blur'].setMinimum(0)
        self.widgets['slider']['blur'].setMaximum(40)
        self.widgets['slider']['blur'].valueChanged.connect(self.blur_value_change)
        self.widgets['slider']['blur'].sliderReleased.connect(self.display_images)
        self.im_trans_params['guass_blur_kern_size'] = 0
        self.widgets['slider']['blur'].setValue(self.im_trans_params['guass_blur_kern_size'])
        self.widgets['label']['blur'] = QLabel(self)
        self.widgets['label']['blur'].setAlignment(Qt.AlignmentFlag.AlignRight)
        self.widgets['label']['blur'].setText('Blur:')
        self.widgets['label']['blur_value'] = QLabel(self)
        self.widgets['label']['blur_value'].setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.widgets['label']['blur_value'].setText(str(self.im_trans_params['guass_blur_kern_size']))
        # brightness
        self.widgets['slider']['brightness'] = QSlider(Qt.Orientation.Horizontal)
        self.widgets['slider']['brightness'].setMinimum(-255)
        self.widgets['slider']['brightness'].setMaximum(255)
        self.widgets['slider']['brightness'].valueChanged.connect(self.brightness_value_change)
        self.widgets['slider']['brightness'].valueChanged.connect(self._display_images_quick)
        self.widgets['slider']['brightness'].sliderReleased.connect(self.display_images)
        self.im_trans_params['brightness_adjustment'] = 0
        self.widgets['slider']['brightness'].setValue(self.im_trans_params['brightness_adjustment'])
        self.widgets['label']['brightness'] = QLabel(self)
        self.widgets['label']['brightness'].setAlignment(Qt.AlignmentFlag.AlignRight)
        self.widgets['label']['brightness'].setText('Brightness:')
        self.widgets['label']['brightness_value'] = QLabel(self)
        self.widgets['label']['brightness_value'].setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.widgets['label']['brightness_value'].setText(str(self.im_trans_params['brightness_adjustment']))
        '''metrics info'''
        self.widgets['label']['real_metrics'] = QLabel(self)
        self.widgets['label']['real_metrics'].setAlignment(Qt.AlignmentFlag.AlignRight)
        self.widgets['label']['real_metrics'].setText('Metrics (real, real_trans):')
        self.widgets['label']['real_metrics_info'] = QLabel(self)
        self.widgets['label']['real_metrics_info'].setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.widgets['label']['real_metrics_info'].setText('')
        self.widgets['label']['gen_metrics'] = QLabel(self)
        self.widgets['label']['gen_metrics'].setAlignment(Qt.AlignmentFlag.AlignRight)
        self.widgets['label']['gen_metrics'].setText('Metrics (sim, gen_trans):')
        self.widgets['label']['gen_metrics_info'] = QLabel(self)
        self.widgets['label']['gen_metrics_info'].setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.widgets['label']['gen_metrics_info'].setText('')
        self.widgets['label']['sim_metrics'] = QLabel(self)
        self.widgets['label']['sim_metrics'].setAlignment(Qt.AlignmentFlag.AlignRight)
        self.widgets['label']['sim_metrics'].setText('Metrics (sim, sim_trans):')
        self.widgets['label']['sim_metrics_info'] = QLabel(self)
        self.widgets['label']['sim_metrics_info'].setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.widgets['label']['sim_metrics_info'].setText('')
        '''error info'''
        self.widgets['label']['real_errors'] = QLabel(self)
        self.widgets['label']['real_errors'].setAlignment(Qt.AlignmentFlag.AlignRight)
        self.widgets['label']['real_errors'].setText('Pose Errors:')
        self.widgets['label']['real_errors_info'] = QLabel(self)
        self.widgets['label']['real_errors_info'].setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.widgets['label']['real_errors_info'].setText('')
        self.widgets['label']['gen_errors'] = QLabel(self)
        self.widgets['label']['gen_errors'].setAlignment(Qt.AlignmentFlag.AlignRight)
        self.widgets['label']['gen_errors'].setText('Pose Errors:')
        self.widgets['label']['gen_errors_info'] = QLabel(self)
        self.widgets['label']['gen_errors_info'].setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.widgets['label']['gen_errors_info'].setText('')
        self.widgets['label']['sim_errors'] = QLabel(self)
        self.widgets['label']['sim_errors'].setAlignment(Qt.AlignmentFlag.AlignRight)
        self.widgets['label']['sim_errors'].setText('Pose Errors:')
        self.widgets['label']['sim_errors_info'] = QLabel(self)
        self.widgets['label']['sim_errors_info'].setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.widgets['label']['sim_errors_info'].setText('')

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
        im_width = 15
        im_height = 15
        button = 1
        slider_width = int(im_width*2)
        check_box_width = 5
        # horizonal start values
        start_im = 1
        start_controls = 0#im_width*2+button
        start_controls_row = (im_height+button)*3+button+start_im
        start_metrics = im_width*2+button

        # display images
        im_row = 0
        for im_type in ['im_real', 'im_gen', 'im_sim']:
            self.layout.addWidget(self.widgets['label'][im_type], start_im-1+im_row*(im_height+button), 0, button, im_width)
            self.layout.addWidget(self.widgets['label'][im_type+'_trans'], start_im-1+im_row*(im_height+button), im_width+button, button, im_width)
            self.layout.addWidget(self.im_Qlabels[im_type], start_im+im_row*(im_height+button), 0,   im_height, im_width)
            self.layout.addWidget(self.im_Qlabels[im_type+'_trans'], start_im+im_row*(im_height+button), im_width+button, im_height, im_width)
            # metircs info
            self.layout.addWidget(self.widgets['label'][im_type[3:]+'_metrics'], start_im+im_row*(im_height+button)+1, (im_width+button)*2, button, button)
            self.layout.addWidget(self.widgets['label'][im_type[3:]+'_metrics_info'], start_im+im_row*(im_height+button)+1, (im_width+button)*2+button, button, button)
            # errors info
            self.layout.addWidget(self.widgets['label'][im_type[3:]+'_errors'], start_im+im_row*(im_height+button)+1, (im_width+button)*2+(button*2), button, button)
            self.layout.addWidget(self.widgets['label'][im_type[3:]+'_errors_info'], start_im+im_row*(im_height+button)+1, (im_width+button)*2+(button*3), button, button)
            im_row += 1

        # load files
        # self.layout.addWidget(self.widgets['button']['load_dataset'], im_height, 1, 1, 1)

        # image buttons (prev, copy, next, etc.)
        self.layout.addWidget(self.widgets['button']['prev'], start_im+im_row*(im_height+button), 1, button, int(im_width*0.66))
        self.layout.addWidget(self.widgets['label']['filename'], start_im+im_row*(im_height+button), int(im_width*0.66)+1, button, int(im_width*0.66))
        self.layout.addWidget(self.widgets['button']['next'], start_im+im_row*(im_height+button), int(im_width*0.66)*2+1, button, int(im_width*0.66))
        # self.layout.addWidget(self.button_copy_im, 0, 1, 1, 1)

        # checkboxes
        i = start_controls_row
        # self.layout.addWidget(self.widgets['checkbox']['real_im'],   button*i, start_controls+button, button, check_box_width)
        # self.layout.addWidget(self.widgets['checkbox']['run_generator'], button*i, start_controls+button+check_box_width, button, check_box_width)
        # i += 1

        # sliders
        sliders = ['rotation', 'blur', 'brightness']
        for slider in sliders:
            self.layout.addWidget(self.widgets['slider'][slider],   button*i, start_controls+button, button, slider_width)
            self.layout.addWidget(self.widgets['label'][slider],    button*i, start_controls,   button, button)
            self.layout.addWidget(self.widgets['label'][slider+'_value'], button*i, start_controls+button+slider_width,   button, button)
            i += 1

        # reset sliders
        self.layout.addWidget(self.widgets['button']['reset_sliders'], button*i, start_controls, button, button)
        self.layout.addWidget(self.widgets['button']['force_update'], button*i, start_controls+button, button, button)
        i += 1
        # init it!
        self.show()

    '''
    ==================== functions to bind to widgets ====================
    '''
    # buttons
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

    # loaders
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

    # checkboxes
    def copy_ref_im(self):
        if self.copy_or_real == 'Real':
            self.copy_or_real = 'Copy'
            self.widgets['checkbox']['run_generator'].setEnabled(False)
        elif self.copy_or_real == 'Copy':
            self.copy_or_real = 'Real'
            self.widgets['checkbox']['run_generator'].setEnabled(True)
        self.load_real_image()
        self.display_images()

    def toggle_generator(self):
        self.run_generator = not self.run_generator
        self.load_real_image()
        self.display_images()

    # sliders
    def rotate_value_change(self):
        self.im_trans_params['angle_to_rotate'] = self.widgets['slider']['rotation'].value()
        self.widgets['label']['rotation_value'].setText(str(self.im_trans_params['angle_to_rotate']))

    def blur_value_change(self):
        self.im_trans_params['guass_blur_kern_size'] = (int(self.widgets['slider']['blur'].value()/2)*2) + 1    # need to make kernel size odd
        if self.im_trans_params['guass_blur_kern_size'] == 1:
            self.im_trans_params['guass_blur_kern_size'] = 0
        self.widgets['label']['blur_value'].setText(str(self.im_trans_params['guass_blur_kern_size']))

    def brightness_value_change(self):
        self.im_trans_params['brightness_adjustment'] = self.widgets['slider']['brightness'].value()
        self.widgets['label']['brightness_value'].setText(str(self.im_trans_params['brightness_adjustment']))
        self.im_trans_params['brightness_adjustment'] = self.im_trans_params['brightness_adjustment']/255

    def reset_sliders(self):
        self.widgets['slider']['rotation'].setValue(0)
        self.widgets['slider']['blur'].setValue(0)
        self.widgets['slider']['brightness'].setValue(0)
        self.display_images()

    '''
    image updaters
    '''
    def load_sim_image(self):
        image = self.df.iloc[self.im_num]['sensor_image']
        self.widgets['label']['filename'].setText(image)
        image_path = os.path.join(self.im_sim_dir, image)
        poses = ['pose_'+str(i) for i in range(1,7)]
        self.sensor_data['poses'] = {}
        for pose in poses:
            self.sensor_data['poses'][pose] = self.df.iloc[self.im_num][pose]
        self.sensor_data['im_sim'] = load_image(image_path)
        self.sensor_data['im_sim'] = gui_utils.process_im(self.sensor_data['im_sim'], data_type='sim')

    def load_real_image(self):
        image_path = os.path.join(self.im_real_dir, self.df.iloc[self.im_num]['sensor_image'])
        self.sensor_data['im_raw'] = gui_utils.load_and_crop_raw_real(image_path)

        self.sensor_data['im_real'] = gui_utils.process_im(self.sensor_data['im_raw'], data_type='real')
        self.sensor_data['im_gen'] = self.generator.get_prediction(self.sensor_data['im_real'])
        # if self.copy_or_real == 'Real':
        #     self.sensor_data['im_compare'] = self.sensor_data['im_real'].copy()
        #     if self.run_generator == True:
        #         self.sensor_data['im_compare'] = self.generator.get_prediction(self.sensor_data['im_compare'])
        # elif self.copy_or_real == 'Copy':
        #     self.sensor_data['im_compare'] = self.sensor_data['im_sim'].copy()

    def transform_image(self, image):
        return gui_utils.transform_image(image, self.im_trans_params)

    def display_images(self):
        change_im(self.im_Qlabels['im_real'], self.sensor_data['im_real'], resize=self.image_display_size)
        im_real_trans = self.transform_image(self.sensor_data['im_real'])
        change_im(self.im_Qlabels['im_real_trans'], im_real_trans, resize=self.image_display_size)
        change_im(self.im_Qlabels['im_gen'], self.sensor_data['im_gen'], resize=self.image_display_size)
        im_gen_trans = self.transform_image(self.sensor_data['im_gen'])
        change_im(self.im_Qlabels['im_gen_trans'], im_gen_trans, resize=self.image_display_size)
        change_im(self.im_Qlabels['im_sim'], self.sensor_data['im_sim'], resize=self.image_display_size)
        im_sim_trans = self.transform_image(self.sensor_data['im_sim'])
        change_im(self.im_Qlabels['im_sim_trans'], im_sim_trans, resize=self.image_display_size)
        self.get_metrics_errors({'real':im_real_trans, 'gen':im_gen_trans, 'sim':im_sim_trans})
        # self.get_metrics_errors(im_real_trans)

    def _display_images_quick(self):
        '''dont update the metrics for speedy interaction'''
        change_im(self.im_Qlabels['im_real'], self.sensor_data['im_real'], resize=self.image_display_size)
        im_real_trans = self.transform_image(self.sensor_data['im_real'])
        change_im(self.im_Qlabels['im_real_trans'], im_real_trans, resize=self.image_display_size)
        change_im(self.im_Qlabels['im_gen'], self.sensor_data['im_gen'], resize=self.image_display_size)
        im_gen_trans = self.transform_image(self.sensor_data['im_gen'])
        change_im(self.im_Qlabels['im_gen_trans'], im_gen_trans, resize=self.image_display_size)
        change_im(self.im_Qlabels['im_sim'], self.sensor_data['im_sim'], resize=self.image_display_size)
        im_sim_trans = self.transform_image(self.sensor_data['im_sim'])
        change_im(self.im_Qlabels['im_sim_trans'], im_sim_trans, resize=self.image_display_size)

    '''
    metrics/error info updaters
    '''
    def get_metrics_errors(self, im_trans_dict):
        metrics = {}
        metrics['real'] = self.metrics.get_metrics(self.sensor_data['im_real'], im_trans_dict['real'])
        metrics['gen'] = self.metrics.get_metrics(self.sensor_data['im_sim'], im_trans_dict['gen'])
        metrics['sim'] = self.metrics.get_metrics(self.sensor_data['im_sim'], im_trans_dict['sim'])

        errors = {}
        errors['real'] = self.pose_esimator_real.get_error(self.sensor_data['im_real'], self.sensor_data['poses'])
        errors['real_trans'] = self.pose_esimator_real.get_error(im_trans_dict['real'], self.sensor_data['poses'])
        errors['gen'] = self.pose_esimator_sim.get_error(self.sensor_data['im_gen'], self.sensor_data['poses'])
        errors['gen_trans'] = self.pose_esimator_sim.get_error(im_trans_dict['gen'], self.sensor_data['poses'])
        errors['sim'] = self.pose_esimator_sim.get_error(self.sensor_data['im_sim'], self.sensor_data['poses'])
        errors['sim_trans'] = self.pose_esimator_sim.get_error(im_trans_dict['sim'], self.sensor_data['poses'])

        for im_type in ['real', 'gen', 'sim']:
            self.display_metrics(metrics[im_type], im_type)
            self.display_errors({'Norm': errors[im_type], 'Trans': errors[im_type+'_trans']}, im_type)

    def display_metrics(self, metrics, label):
        text = ''
        for key in metrics.keys():
            metric = str(metrics[key][0])
            metric = metric[:min(len(metric), 5)]
            text += key + ': ' + metric + '\n'
        self.widgets['label'][label+'_metrics_info'].setText(text)

    def display_errors(self, errors_dict, label):
        text = ''
        for key in errors_dict.keys():
            text += key + ': ' + str(errors_dict[key]['MAE'][0])[:5] + '\n'
        self.widgets['label'][label+'_errors_info'].setText(text)

    '''
    utils
    '''
    def load_dataset(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.im_num = 0   # row of csv dataset to use
        self.im_sim_dir = os.path.join(os.path.dirname(csv_file), 'images')
        self.im_real_dir = os.path.join(os.path.dirname(gui_utils.get_real_csv_given_sim(csv_file)), 'images')
        self.load_sim_image()
        self.load_real_image()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='image file to use', default=os.path.join(os.path.expanduser('~'),'summer-project/data/Bourne/tactip/sim/surface_3d/tap/128x128/csv_train/images/image_1.png'))
    parser.add_argument('--csv_path', type=str, help='csv file to use', default=os.path.join(os.path.expanduser('~'),'summer-project/data/Bourne/tactip/sim/surface_3d/shear/128x128/csv_train/targets.csv'))
    parser.add_argument('--generator_path', type=str, help='generator weights file to use', default=os.path.join(os.path.expanduser('~'),'summer-project/models/sim2real/matt/surface_3d/shear/pretrained_edge_tap/no_ganLR:0.0002_decay:0.1_BS:64_DS:1.0/run_0/models/best_generator.pth'))
    parser.add_argument('--pose_path', type=str, help='generator weights file to use', default=os.path.join(os.path.expanduser('~'), 'summer-project/models/pose_estimation/surface_3d/shear/sim_LR:0.0001_BS:16/run_0/checkpoints/best_model.pth'))
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = make_app(app, args)
    sys.exit(app.exec())
