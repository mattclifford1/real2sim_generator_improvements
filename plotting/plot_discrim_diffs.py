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
    parser.add_argument("--csv", default='results/compare_discriminators.csv', help='path to csv file')
    parser.add_argument("--metric", default='Score on discriminator (accuracy)')
    ARGS = parser.parse_args()

    results = results_reader(csv_file=ARGS.csv)
    groupby = 'Generator'
    inner_groupby = 'Discriminator'
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


    # plot grouped bar chart
    plot_df.plot(x=groupby,
            kind='bar',
            stacked=False,
            # title='Average Dicriminator Score with'
            )
    plt.ylabel(ARGS.metric)
    plt.xticks(rotation=0)
    plt.legend(loc='right', title=inner_groupby)
    save_file = 'results/diff_discrim.png'
    plt.savefig(save_file)
    print('Saved figure to: '+save_file)
