import matplotlib.pyplot as plt
import pandas as pd
import ast

class results_reader():
    def __init__(self, csv_file='results/compare_existing_models.csv'):
        self.get_df()

    def get_df(self, csv_file='results/compare_existing_models.csv'):
        self.df = pd.read_csv(csv_file, index_col=0)
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
    results = results_reader()
    groupby = 'GEN'
    inner_groupby = 'DATA'
    plot_order = ['edge_tap', 'edge_shear', 'surface_tap', 'surface_shear']
    plot_data = []
    for plot in plot_order:
        group_data = [plot]
        exp_order = []
        plots = 0
        for result in results:
            if result[groupby] == plot:
                group_data.append(result['MSE on Validation'])
                exp_order.append(result[inner_groupby])
                plots += 1
            if plots == 4:  # hack to not plot repeated resutls
                break
        print(exp_order)  # make sure all in right order
        plot_data.append(group_data)
    plot_df = pd.DataFrame(plot_data, columns=[groupby]+exp_order)


    # plot grouped bar chart
    plot_df.plot(x=groupby,
            kind='bar',
            stacked=False,
            title='Grouped Bar Graph with differing '+inner_groupby)
    plt.show()
