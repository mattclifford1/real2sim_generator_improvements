import pandas as pd
import matplotlib.pyplot as plt
import os


dir = os.path.join('image_transformations', 'results_surface_val')
results = {}
for file in os.listdir(dir):
    if os.path.splitext(file)[1] == '.csv':
        name = file.split('_')[0]
        if len(name) == 1:
            name += '_shift'
        results[name] = pd.read_csv(os.path.join(dir, file))
        results[name]['MAE x10'] = results[name]['MAE']*10
        results[name]['MSE x10'] = results[name]['MSE']*10


fig, axs = plt.subplots(nrows=2, ncols=len(results.keys())//2)
# fig, ax = plt.subplots(nrows=1, ncols=len(results.keys()))
for key, col in zip(results.keys(), axs.reshape(-1)):
    if key != 'brightness':
        results[key].plot.line(x=key, y=['PoseNet', 'SSIM', 'MSSIM', 'NLPD', 'MSE x10', 'MAE x10', 'LPIPS_vgg'], ax=col)
    else:
        results[key].plot.line(x=key, y=['PoseNet', 'SSIM', 'MSSIM', 'NLPD', 'MSE', 'MAE', 'LPIPS_vgg'], ax=col)
    if key in ['brightness', 'y_shift', 'blur']:
        col.legend(loc='upper left')
    else:
        col.legend(loc='upper right')
    # col.set_ylabel('Metrics')
plt.show()
