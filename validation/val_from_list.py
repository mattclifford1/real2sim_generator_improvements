import os
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import sys; sys.path.append('..'); sys.path.append('.')
from validation.val_all_metrics import run_val

def save_df(results, task, location='validation/models_evaluation_on_'):
    df = pd.DataFrame.from_dict(results)
    df.to_csv(location+task[0]+'_'+task[1]+'.csv', index=False)



parser = ArgumentParser(description='')
parser.add_argument("--dir", default='..', help='path to folder where training graphs are within')
parser.add_argument("--batch_size",type=int,  default=32, help='batch size to load and train on')
parser.add_argument("--ram", default=False, action='store_true', help='load dataset into ram')
ARGS = parser.parse_args()


# define the  label:filepath   to val
base_dir = os.path.join(ARGS.dir, 'models', 'sim2real', 'matt')
pretrained = 'pretrained_edge_tap'
train_routine = 'LR:0.0002_decay:0.1_BS:64'
data_limits = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
run = 'run_1'
models = {
    # '''not pretrained - limited data'''
    'surface shear '+str(data_limits[0]): os.path.join(base_dir, 'surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[0]), run, 'models', 'best_generator.pth'),
    'surface shear '+str(data_limits[1]): os.path.join(base_dir, 'surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[1]), run, 'models', 'best_generator.pth'),
    'surface shear '+str(data_limits[2]): os.path.join(base_dir, 'surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[2]), run, 'models', 'best_generator.pth'),
    'surface shear '+str(data_limits[3]): os.path.join(base_dir, 'surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[3]), run, 'models', 'best_generator.pth'),
    'surface shear '+str(data_limits[4]): os.path.join(base_dir, 'surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[4]), run, 'models', 'best_generator.pth'),
    'surface shear '+str(data_limits[5]): os.path.join(base_dir, 'surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[5]), run, 'models', 'best_generator.pth'),
    'surface shear '+str(data_limits[6]): os.path.join(base_dir, 'surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[6]), run, 'models', 'best_generator.pth'),
    'surface shear '+str(data_limits[7]): os.path.join(base_dir, 'surface_3d', 'shear', 'not_pretrained', 'no_gan'+train_routine+'_DS:'+str(data_limits[7]), run, 'models', 'best_generator.pth'),
    #
    # # '''pretrained - limited data'''
    '[pre-et]surface shear '+str(data_limits[0]): os.path.join(base_dir, 'surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[0]), run, 'models', 'best_generator.pth'),
    '[pre-et]surface shear '+str(data_limits[1]): os.path.join(base_dir, 'surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[1]), run, 'models', 'best_generator.pth'),
    '[pre-et]surface shear '+str(data_limits[2]): os.path.join(base_dir, 'surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[2]), run, 'models', 'best_generator.pth'),
    '[pre-et]surface shear '+str(data_limits[3]): os.path.join(base_dir, 'surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[3]), run, 'models', 'best_generator.pth'),
    '[pre-et]surface shear '+str(data_limits[4]): os.path.join(base_dir, 'surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[4]), run, 'models', 'best_generator.pth'),
    '[pre-et]surface shear '+str(data_limits[5]): os.path.join(base_dir, 'surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[5]), run, 'models', 'best_generator.pth'),
    '[pre-et]surface shear '+str(data_limits[6]): os.path.join(base_dir, 'surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[6]), run, 'models', 'best_generator.pth'),
    '[pre-et]surface shear '+str(data_limits[7]): os.path.join(base_dir, 'surface_3d', 'shear', pretrained, 'no_gan'+train_routine+'_DS:'+str(data_limits[7]), run, 'models', 'best_generator.pth'),

    'edge tap [pre-net] ': os.path.join(ARGS.dir, 'models/sim2real/alex/trained_gans/[edge_2d]/128x128_[tap]_250epochs/checkpoints/best_generator.pth'),
}

task = ['surface_3d', 'shear']
results = {}
init_keys = False
for key in tqdm(models.keys(), desc='All models'):
    stats = run_val(ARGS.dir, task, ARGS.batch_size, models[key], ram=ARGS.ram)
    if init_keys is False:
        results['name'] = [key]
        for k in stats.keys():
            results[k] = [stats[k]]
        init_keys = True
    else:
        results['name'].append(key)
        for k in stats.keys():
            results[k].append(stats[k])
    save_df(results, task)
