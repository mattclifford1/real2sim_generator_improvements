import os
import itertools
import shutil
from tqdm import tqdm
import cv2
import sys; sys.path.append('..'); sys.path.append('.')
from trainers.image_transforms import process_image, process_image_sim
from trainers.data_loader import get_params, get_all_ims
from trainers.utils import check_task


class task_job():
    def __init__(self,
                 base_dir='..',
                 data='data/Bourne/tactip',
                 task=('edge_2d', 'tap'),
                 size=128,
                 val=True,
                 store_ram=False,
                 use_percentage_of_data=None):
        # real_images_dir = os.path.join(dir, 'data/Bourne/tactip/real/'+data[0]+'/'+data[1]+'/csv_val/images')
        check_task(task)
        self.base_dir = base_dir
        self.dir_structure = data
        self.dir = os.path.join(base_dir, data)
        self.task = task
        self.split_type = 'csv_val' if val else 'csv_train'
        self.real_dir = os.path.join(self.dir_structure, 'real', task[0], task[1], self.split_type, 'images')
        self.sim_dir = os.path.join(self.dir_structure, 'sim', task[0], task[1], str(size)+'x'+str(size), self.split_type, 'images')
        self.real_meta_dir = os.path.join(self.dir_structure, 'real', task[0], task[1], self.split_type)
        self.sim_meta_dir = os.path.join(self.dir_structure, 'sim', task[0], task[1], str(size)+'x'+str(size), self.split_type)
        self.images = get_all_ims(os.path.join(self.base_dir, self.real_dir))
        self.im_params = get_params(True)
        self.im_params['normlz'] = False
        self.im_params['size'] = (size, size)

    def process_dataset(self):
        os.makedirs(self.real_dir, exist_ok=True)
        os.makedirs(self.sim_dir, exist_ok=True)
        # copy meta data csv file
        for file in ['targets.csv', 'meta.json']:
            for data_dir in [self.real_meta_dir, self.sim_meta_dir]:
                dst = os.path.join(data_dir, file)
                if not os.path.exists(dst):
                    shutil.copyfile(os.path.join(self.base_dir, data_dir, file), dst)

        for i in tqdm(range(len(self.images)), desc='Loop dataset', leave=False):

            real_image_filename = os.path.join(self.base_dir, self.real_dir, self.images[i])
            sim_image_filename = os.path.join(self.base_dir, self.sim_dir, self.images[i])

            real_write_file = os.path.join(self.real_dir, self.images[i])
            sim_write_file = os.path.join(self.sim_dir, self.images[i])

            if not os.path.exists(real_write_file):
                real = cv2.imread(real_image_filename)
                real = process_image(real, self.im_params)
                cv2.imwrite(real_write_file, real)

            if not os.path.exists(sim_write_file):
                sim = cv2.imread(sim_image_filename)
                sim = process_image_sim(sim, self.im_params)
                cv2.imwrite(os.path.join(self.sim_dir, self.images[i]), sim)



if __name__ == '__main__':
    val = [False, True]
    task = ['edge_2d','surface_3d']
    sampling = ['tap', 'shear']
    datas = list(itertools.product(val, task, sampling))

    for data in tqdm(datas):
        job = task_job(val=data[0], task=[data[1], data[2]])
        job.process_dataset()
