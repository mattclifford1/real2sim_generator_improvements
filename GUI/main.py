import sys
import argparse
from functools import partial
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
        # self.image_display_size = (200, 200)
        # self.image_display_size = (128, 128)
        self.image_display_size = (175, 175)
        self.display_images()
        self.reset_sliders()

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
            self.sensor_data['Xs'] = np.zeros([128, 128, 1], dtype=np.uint8)

        # self.widgets['image']['colour_output'].mousePressEvent = self.image_click
        # hold video frames
        # dummy_frame = [self.dummy_im]*2
        # self.input_frames = {'UI_colour_original':dummy_frame,
                             # 'UI_depth':dummy_frame}
        # self.num_frames = len(self.input_frames['UI_colour_original'])

    def init_widgets(self):
        '''
        create all the widgets we need and init params
        '''
        # define what sliders we are using
        self.sliders = {
                   'rotation':{'min':-180, 'max':180, 'init_value':0, 'value_change':[partial(self.generic_value_change, 'rotation', normalise=None), self.display_images], 'release': [self.display_images]},
                   'blur':{'min':0, 'max':40, 'init_value':0, 'value_change':[partial(self.generic_value_change, 'blur', normalise='odd')], 'release': [self.display_images]},
                   'brightness':{'min':-255, 'max':255, 'init_value':0, 'value_change':[partial(self.generic_value_change, 'brightness', normalise=255)], 'release': [self.display_images]},
                   'zoom':{'min':10, 'max':400, 'init_value':100, 'value_change':[partial(self.generic_value_change, 'zoom', normalise=100), self.display_images], 'release': [self.display_images]},
                   'x_shift':{'min':-100, 'max':100, 'init_value':0, 'value_change':[partial(self.generic_value_change, 'x_shift', normalise=100), self.display_images], 'release': [self.display_images]},
                   'y_shift':{'min':-100, 'max':100, 'init_value':0, 'value_change':[partial(self.generic_value_change, 'y_shift', normalise=100), self.display_images], 'release': [self.display_images]},
                   }

        '''images'''
        # set up layout of images
        self.im_pair_names = [
                              ('Xr', 'T(Xr)'),
                              ('Xs', 'T(Xs)'),
                              ('G(Xr)', 'T(G(Xr))'),
                              ('G(T(Xr))', 'T(G(Xr))_'),
                              ]
        # widget dictionary store
        self.widgets = {'button': {}, 'slider': {}, 'checkbox': {}, 'label': {}, 'image':{}}
        for im_pair in self.im_pair_names:
            for im_name in im_pair:
                # image widget
                self.widgets['image'][im_name] = QLabel(self)
                self.widgets['image'][im_name].setAlignment(Qt.AlignmentFlag.AlignCenter)
                # image label
                self.widgets['label'][im_name] = QLabel(self)
                self.widgets['label'][im_name].setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.widgets['label'][im_name].setText(im_name)
                # self.widgets['label'][im_name].setContentsMargins(0,0,0,0)
            # metrics info
            self.widgets['label'][str(im_pair)+'_metrics'] = QLabel(self)
            self.widgets['label'][str(im_pair)+'_metrics'].setAlignment(Qt.AlignmentFlag.AlignRight)
            self.widgets['label'][str(im_pair)+'_metrics'].setText('Metrics '+str(im_pair)+':')
            self.widgets['label'][str(im_pair)+'_metrics_info'] = QLabel(self)
            self.widgets['label'][str(im_pair)+'_metrics_info'].setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.widgets['label'][str(im_pair)+'_metrics_info'].setText('')
            # error info
            self.widgets['label'][str(im_pair)+'_errors'] = QLabel(self)
            self.widgets['label'][str(im_pair)+'_errors'].setAlignment(Qt.AlignmentFlag.AlignRight)
            self.widgets['label'][str(im_pair)+'_errors'].setText('Pose Errors:')
            self.widgets['label'][str(im_pair)+'_errors_info'] = QLabel(self)
            self.widgets['label'][str(im_pair)+'_errors_info'].setAlignment(Qt.AlignmentFlag.AlignRight)
            self.widgets['label'][str(im_pair)+'_errors_info'].setText('')

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

        '''sliders'''
        self.im_trans_params = {}
        for key in self.sliders.keys():
            self.widgets['slider'][key] = QSlider(Qt.Orientation.Horizontal)
            self.widgets['slider'][key].setMinimum(self.sliders[key]['min'])
            self.widgets['slider'][key].setMaximum(self.sliders[key]['max'])
            for func in self.sliders[key]['value_change']:
                self.widgets['slider'][key].valueChanged.connect(func)
            for func in self.sliders[key]['release']:
                self.widgets['slider'][key].sliderReleased.connect(func)
            self.im_trans_params[key] = self.sliders[key]['init_value']
            self.widgets['label'][key] = QLabel(self)
            self.widgets['label'][key].setAlignment(Qt.AlignmentFlag.AlignRight)
            self.widgets['label'][key].setText(key+':')
            self.widgets['label'][key+'_value'] = QLabel(self)
            self.widgets['label'][key+'_value'].setAlignment(Qt.AlignmentFlag.AlignRight)
            self.widgets['label'][key+'_value'].setText(str(self.im_trans_params[key]))

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
        for im_pair in self.im_pair_names:
            self.layout.addWidget(self.widgets['label'][im_pair[0]], start_im-1+im_row*(im_height+button), 0, button, im_width)
            self.layout.addWidget(self.widgets['label'][im_pair[1]], start_im-1+im_row*(im_height+button), im_width+button, button, im_width)
            self.layout.addWidget(self.widgets['image'][im_pair[0]], start_im+im_row*(im_height+button), 0,   im_height, im_width)
            self.layout.addWidget(self.widgets['image'][im_pair[1]], start_im+im_row*(im_height+button), im_width+button, im_height, im_width)
            # metircs info
            self.layout.addWidget(self.widgets['label'][str(im_pair)+'_metrics'], start_im+im_row*(im_height+button)+1, (im_width+button)*2, button, button)
            self.layout.addWidget(self.widgets['label'][str(im_pair)+'_metrics_info'], start_im+im_row*(im_height+button)+1, (im_width+button)*2+button, im_height, button)
            # errors info
            self.layout.addWidget(self.widgets['label'][str(im_pair)+'_errors'], start_im+im_row*(im_height+button)+1, (im_width+button)*2+(button*2), button, button)
            self.layout.addWidget(self.widgets['label'][str(im_pair)+'_errors_info'], start_im+im_row*(im_height+button)+1, (im_width+button)*2+(button*3), im_height, button)
            im_row += 1

        # load files
        # self.layout.addWidget(self.widgets['button']['load_dataset'], im_height, 1, 1, 1)

        # image buttons (prev, copy, next, etc.)
        self.layout.addWidget(self.widgets['button']['prev'], start_im+im_row*(im_height+button), 1, button, int(im_width*0.66))
        self.layout.addWidget(self.widgets['label']['filename'], start_im+im_row*(im_height+button), int(im_width*0.66)+1, button, int(im_width*0.66))
        self.layout.addWidget(self.widgets['button']['next'], start_im+im_row*(im_height+button), int(im_width*0.66)*2+1, button, int(im_width*0.66))
        # self.layout.addWidget(self.button_copy_im, 0, 1, 1, 1)

        i = (im_height+button)*im_row+button+start_im
        # checkboxes
        # self.layout.addWidget(self.widgets['checkbox']['real_im'],   button*i, start_controls+button, button, check_box_width)
        # self.layout.addWidget(self.widgets['checkbox']['run_generator'], button*i, start_controls+button+check_box_width, button, check_box_width)
        # i += 1

        # sliders
        for slider in self.sliders.keys():
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

    # sliders value changes
    def generic_value_change(self, key, normalise=None):
        if normalise == 'odd':
            self.im_trans_params[key] = (int(self.widgets['slider'][key].value()/2)*2) + 1    # need to make kernel size odd
            if self.im_trans_params[key] == 1:
                self.im_trans_params[key] = 0
        else:
            self.im_trans_params[key] = self.widgets['slider'][key].value()
        if type(normalise) is int:
            self.im_trans_params[key] = self.im_trans_params[key]/normalise
        # display the updated value
        value_str = str(self.im_trans_params[key])
        disp_len = 4
        if len(value_str) > disp_len:
            value_str = value_str[:disp_len]
        elif len(value_str) < disp_len:
            value_str = ' '*(disp_len-len(value_str)) + value_str
        self.widgets['label'][key+'_value'].setText(value_str)

    def reset_sliders(self):
        for key in self.sliders.keys():
            self.widgets['slider'][key].setValue(self.sliders[key]['init_value'])
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
        self.sensor_data['Xs'] = load_image(image_path)
        self.sensor_data['Xs'] = gui_utils.process_im(self.sensor_data['Xs'], data_type='sim')

    def load_real_image(self):
        image_path = os.path.join(self.im_real_dir, self.df.iloc[self.im_num]['sensor_image'])
        self.sensor_data['im_raw'] = gui_utils.load_and_crop_raw_real(image_path)

        self.sensor_data['Xr'] = gui_utils.process_im(self.sensor_data['im_raw'], data_type='real')
        self.sensor_data['G(Xr)'] = self.generator.get_prediction(self.sensor_data['Xr'])

    def transform_image(self, image):
        return gui_utils.transform_image(image, self.im_trans_params)

    def display_images(self):
        self._display_images_quick()
        self.get_metrics_errors()

    def _display_images_quick(self):
        # get transformed images
        self.sensor_data['T(Xr)'] = self.transform_image(gui_utils.process_im(self.sensor_data['im_raw'], data_type='real'))
        self.sensor_data['G(T(Xr))'] = self.generator.get_prediction(self.sensor_data['T(Xr)'])
        self.sensor_data['T(G(Xr))'] = self.transform_image(self.sensor_data['G(Xr)'])
        self.sensor_data['T(G(Xr))_'] = self.sensor_data['T(G(Xr))']
        self.sensor_data['T(Xs)'] = self.transform_image(self.sensor_data['Xs'])

        # display images
        for im_pair in self.im_pair_names:
            for im_name in im_pair:
                change_im(self.widgets['image'][im_name], self.sensor_data[im_name], resize=self.image_display_size)

    '''
    metrics/error info updaters
    '''
    def get_metrics_errors(self):
        metrics = {}
        errors = {}
        for im_pair in self.im_pair_names:
            metrics[str(im_pair)] = self.metrics.get_metrics(self.sensor_data[im_pair[0]], self.sensor_data[im_pair[1]])
            errors[str(im_pair)] = {}
            for im_name in im_pair:
                # decide whether to use real or simulated space pose estimation
                if 'Xs' in im_name or 'G' in im_name:
                    errors[str(im_pair)][im_name] = self.pose_esimator_sim.get_error(self.sensor_data[im_name], self.sensor_data['poses'])
                else:
                    errors[str(im_pair)][im_name] = self.pose_esimator_real.get_error(self.sensor_data[im_name], self.sensor_data['poses'])

            self.display_metrics(metrics[str(im_pair)], str(im_pair))
            self.display_errors(errors[str(im_pair)], str(im_pair))

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
    parser.add_argument('--pose_path', type=str, help='pose net weights file to use', default=os.path.join(os.path.expanduser('~'), 'summer-project/models/pose_estimation/surface_3d/shear/sim_LR:0.0001_BS:16/run_0/checkpoints/best_model.pth'))
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = make_app(app, args)
    sys.exit(app.exec())
