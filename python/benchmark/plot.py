import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='OPT Plot')
    parser.add_argument('-f', '--file',type=str,
                        help='Path to the parquet file')

    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    df = df.replace({'objective_name': 'FairSVM[C=1.0,obj=0,rho=0.01]'}, 'FairSVM: constraints')
    df = df.replace({'objective_name': 'FairSVM[C=1.0,obj=1,rho=0.01]'}, 'FairSVM: objective')

    df = df[df['objective_value'] < 1.0]
    # df = df[df['objective_name'] == 'FairSVM: objective']
    df = df[df['data_name'] != 'Simulated[n_features=100,n_samples=500000]']

    solvers = list(set(df['solver_name']) - set(['rehline']))
    # f, ax = plt.subplots(figsize=(7, 7))
    # ax.set(xscale="log", yscale="log")
    ax = sns.lmplot(data=df, x='time', y='objective_value', 
                row='data_name',
                col='objective_name', hue='solver_name',
                fit_reg=False, hue_order=['rehline'] + solvers,
                legend=True, truncate=True, height=5,
                facet_kws={"sharex":True,"sharey":False}, 
                scatter_kws={"s": 10, "alpha": 0.4})
    ax.set(xscale='log', yscale='log')
    sns.move_legend(ax, "upper right", bbox_to_anchor=(.95, .95))
    # plt.ylim(0, 2.0)
    plt.tight_layout()
    # plt.legend(loc='best')
    plt.show()

# benchopt_run_2023-04-27_09h19m17.parquet