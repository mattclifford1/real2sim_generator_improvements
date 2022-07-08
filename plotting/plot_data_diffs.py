import matplotlib.pyplot as plt
import pandas as pd
import ast
from argparse import ArgumentParser

class results_reader():
    def __init__(self, csv_file='results/compare_existing_models.csv'):
        self.csv_file = csv_file
        self.get_df()

    def get_df(self):
        self.df = pd.read_csv(self.csv_file, index_col=0)
        self.col_names = self.df.columns.to_list()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        deets = ast.literal_eval(row.name)
        for col in self.col_names:
            deets[col] = row[col]
        return deets

if __name__ == '__main__':
    parser = ArgumentParser(description='Test data with GAN models')
    parser.add_argument("--csv", default='results/compare_generators_with_data.csv', help='path to csv file')
    parser.add_argument("--metric", default='MSE', help='metric to use: MSE or SSIM')
    ARGS = parser.parse_args()

    results = results_reader(csv_file=ARGS.csv)
    groupby = 'Generator'
    inner_groupby = 'Data'
    plot_order = ['edge_tap', 'edge_shear', 'surface_tap', 'surface_shear']
    plot_data = []
    for plot in plot_order:
        group_data = [plot]
        exp_order = []
        plots = 0
        for result in results:
            if result[groupby] == plot:
                group_data.append(result[ARGS.metric])
                exp_order.append(result[inner_groupby])
                plots += 1
        plot_data.append(group_data)
    plot_df = pd.DataFrame(plot_data, columns=[groupby]+exp_order)
    print(plot_df)
    # plot grouped bar chart
    plot_df.plot(x=groupby,
            kind='bar',
            stacked=False,
            # title=ARGS.metric+' with Differing '+inner_groupby
            )
    plt.ylabel(ARGS.metric)
    plt.xticks(rotation=0)
    plt.legend(loc='right', title=inner_groupby)
    values_with_gen_strs = plot_df.to_numpy()[:, 1:]
    min_value = values_with_gen_strs.min() - 0.1
    if min_value > 0:
        plt.ylim(min_value)
    save_file = 'results/diff_data_'+ARGS.metric+'.png'
    plt.savefig(save_file)
    print('Saved figure to: '+save_file)
