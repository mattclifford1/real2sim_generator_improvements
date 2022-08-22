import pandas as pd
import matplotlib.pyplot as plt
import os


dir = os.path.join('image_transformations', 'results_val')
results = {}
for file in os.listdir(dir):
    if os.path.splitext(file)[1] == '.csv':
        name = file.split('_')[0]
        if len(name) == 1:
            name += '_shift'
        results[name] = pd.read_csv(os.path.join(dir, file))
        results[name]['MAE x10'] = results[name]['MAE']*10
        results[name]['MSE x10'] = results[name]['MSE']*10

for key in results.keys():
    if key != 'brightness':
        results[key].plot.line(x=key, y=['PoseNet', 'SSIM', 'MSSIM', 'NLPD', 'MSE x10', 'MAE x10', 'LPIPS_vgg'])
    else:
        results[key].plot.line(x=key, y=['PoseNet', 'SSIM', 'MSSIM', 'NLPD', 'MSE', 'MAE', 'LPIPS_vgg'])
    plt.ylabel('Metrics')
    plt.legend(loc='upper right')
    plt.show()
