import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

eps = 1e-5

for file_name in ['fairsvm_out.csv', 'qr_out.csv', 'huber_out.csv', 'svm_out.csv']:

    perf = {'objective_name': [], 
        'data_name': [], 
        'solver': [], 
        'if_solve': [],
        'time': []}

    df = pd.read_csv(file_name)

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
                else:
                    perf['objective_name'].append(obj_name)
                    perf['data_name'].append(data_name)
                    perf['solver'].append(solver)
                    perf['time'].append(np.nan)
                    perf['if_solve'].append(False)

    perf = pd.DataFrame(perf)
    perf = perf.pivot(index=['objective_name', 'data_name'], columns='solver', values='time')
    print(perf.to_markdown())
    print('\n')