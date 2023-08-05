import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def clean_res(res):
    res = pd.DataFrame(res)
    res['objective_name'] = res['objective_name'].replace('FairSVM[C=1.0,obj=1,rho=0.01]', 'FairSVM-obj')
    res['objective_name'] = res['objective_name'].replace('FairSVM[C=1.0,obj=0,rho=0.01]', 'FairSVM-C')
    res = res.replace({'objective_name': r'\[.*.\]'}, {'objective_name': ''}, regex=True)

    # for col_tmp in ['data_name', 'solver']:
    for col_tmp in ['data_name']:
        res = res.replace({col_tmp: r'.*.*\='}, {col_tmp: ''}, regex=True)
        res = res.replace({col_tmp: r'\]'}, {col_tmp: ''}, regex=True)
    return res

eps = 1e-5
pd.reset_option('display.float_format')
res = {'objective_name': [], 
        'data_name': [], 
        'solver': [], 
        'if_solve': [],
        'time': []}

for file_name in ['fairsvm_out.csv', 'qr_out.csv', 'huber_out.csv', 'svm_out.csv']:

    df = pd.read_csv(file_name)
    # df = pd.read_parquet
    perf = {'objective_name': [], 
        'data_name': [], 
        'solver': [], 
        'if_solve': [],
        'time': []}

    dataset_lst = set(df['data_name'])

    for obj_name in set(df['objective_name']):
        for data_name in set(df['data_name']):
            opt_obj = min(df[(df['objective_name'] == obj_name) & (df['data_name'] == data_name)]['objective_value'])
            for solver in set(df['solver_name']):
                dt = df[(df['objective_name'] == obj_name) & (df['data_name'] == data_name) & (df['solver_name'] == solver)]
                dt = dt[(dt['objective_value'] - opt_obj) < eps*np.maximum(dt['objective_value'], 1)]['time']
                if len(dt) > 0:
                    time_tmp = min(dt)
                    
                    perf['objective_name'].append(obj_name)
                    perf['data_name'].append(data_name)
                    perf['solver'].append(solver)
                    perf['time'].append(time_tmp)
                    perf['if_solve'].append(True)

                    res['objective_name'].append(obj_name)
                    res['data_name'].append(data_name)
                    res['solver'].append(solver)
                    res['time'].append(time_tmp)
                    res['if_solve'].append(True)
                else:
                    perf['objective_name'].append(obj_name)
                    perf['data_name'].append(data_name)
                    perf['solver'].append(solver)
                    perf['time'].append(np.nan)
                    perf['if_solve'].append(False)

                    res['objective_name'].append(obj_name)
                    res['data_name'].append(data_name)
                    res['solver'].append(solver)
                    res['time'].append(np.nan)
                    res['if_solve'].append(False)

    perf = clean_res(perf)
    perf.data_name = pd.Categorical(perf.data_name, categories=['steel-plates-fault', 'philippine', 'sylva_prior', 'creditcard',
                                         'liver-disorders', 'kin8nm', 'house_8L', 'topo_2_1', 'Buzzinsocialmedia_Twitter'])
    perf = perf.pivot(index=['objective_name', 'data_name'], columns='solver', values='time')
    print(perf.to_markdown(floatfmt='10.3E'))
    print('\n')
    for col_tmp in perf.columns:
        print('speed-up of rehline/%s: min: %.1f - max: %.1f' %(col_tmp, np.nanmin(perf[col_tmp]/perf['rehline[shrink=True]']), np.nanmax(perf[col_tmp]/perf['rehline[shrink=True]'])))
    print('\n')


res = clean_res(res)
res['objective_name'] = res['objective_name'].replace('ElasticHuber', 'RidgeHuber')
res.data_name = pd.Categorical(res.data_name, categories=['steel-plates-fault', 'philippine', 'sylva_prior', 'creditcard',
                                         'liver-disorders', 'kin8nm', 'house_8L', 'topo_2_1', 'Buzzinsocialmedia_Twitter'])
res['data_name'] = res.data_name.astype(str)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

solvers = list(set(res['solver']) - set(['rehline']))

ax = sns.catplot(x="data_name", y="time", hue='solver', row="objective_name", data=res, kind="bar",
                    hue_order=['rehline'] + solvers, sharex=False, sharey=False, height=3,
                    palette='muted')
ax.set(yscale='log')
# sns.move_legend(ax, "upper right", bbox_to_anchor=(.95, .95))
plt.tight_layout()
plt.show()
