import matplotlib.pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Plot training graphs')
    parser.add_argument("--dir", default='..', help='path to folder where training graphs are within')
    ARGS = parser.parse_args()

    list_of_files = []
    for (dirpath, dirnames, filenames) in os.walk(ARGS.dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                list_of_files.append(os.sep.join([dirpath, filename]))


    cols = ['mean training loss', 'val MSE', 'val_SSIM']
    fig, ax = plt.subplots(nrows=1, ncols=len(cols))

    for i, col in enumerate(ax):
        for file in list_of_files:
            df = pd.read_csv(file)
            # print(df['epoch'].values)
            label = file.split('/')[-3]
            col.plot(df['epoch'].values[1:], df[cols[i]].values[1:], label=label)

        col.legend()
        col.set_title(cols[i])
    plt.show()
