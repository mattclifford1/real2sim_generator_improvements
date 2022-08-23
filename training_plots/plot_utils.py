import os
import pandas as pd
import numpy as np

def get_avg_of_runs(dir, col_to_get='Downstream MAE', csv_file='training_stats.csv'):
    Ys = []
    for run in os.listdir(dir):
        path = os.path.join(dir, run, csv_file)
        df = pd.read_csv(path)
        if len(df) == 251:
            Ys.append(df[col_to_get].values[0:])
            x = df['epoch'].values[0:]
    Ys = np.array(Ys)
    return x, np.mean(Ys, axis=0), np.std(Ys, axis=0)
