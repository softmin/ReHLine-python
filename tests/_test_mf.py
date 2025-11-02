'''Test MF on simulated dataset'''
import numpy as np
from rehline import plqMF_Ridge, make_mf_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import itertools


## Data Preparation
user_num, item_num = 1200, 4000
ratings = make_mf_dataset(n_users=user_num, n_items=item_num, n_interactions=50000, seed=42)
X_train, X_test, y_train, y_test = train_test_split(ratings['X'], ratings['y'], test_size=0.3, random_state=42)
n_jobs = -1
verbose = 0



## Function for Parallel Grid Search
def parallel_grid_search(model_class, param_grid, fixed_params,  X_train, y_train, X_val, y_val, n_jobs=-1):
    """
    Perform parallel grid search to find the optimal hyperparameters for a given model class.
    
    This function evaluates all possible combinations of hyperparameters specified in the 
    parameter grid using parallel processing to accelerate the search process. Each parameter 
    combination is used to train a model and evaluate its performance on validation data.
    
    Parameters:
    -----------
    model_class : class
        The model class to be instantiated and evaluated.
    param_grid : dict
        Dictionary containing hyperparameters to search over. Keys are parameter names 
        and values are lists of possible parameter values.
    fixed_params : dict
        Dictionary containing fixed hyperparameters to make sure model works. Keys are parameter names 
        and values are possible parameter values.
    X_train : array-like
        Training feature data.
    y_train : array-like
        Training target data.
    X_val : array-like
        Validation feature data for model evaluation.
    y_val : array-like
        Validation target data for model evaluation.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 means using all available processors.
    
    Returns:
    --------
    best_params : dict
        Dictionary containing the best performing hyperparameters.
    best_score : float
        The best score achieved by the optimal hyperparameters.
    all_results : list
        List of dictionaries containing all evaluated parameter combinations and their scores.
    
    Notes:
    ------
    - The function uses the model's `obj` method for scoring and normalizes by `n_ratings`.
    - Lower scores are considered better (minimization problem).
    """
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    #print(f"Testing {len(param_combinations)} parameter combinations")

    def evaluate_single_params(params):
        param_dict = dict(zip(param_names, params))
        model = model_class(**param_dict, **fixed_params)
        model.fit(X_train, y_train)

        accuracy = None
        if len(np.unique(y_train)) == 2: # classification
            y_pred = model.decision_function(X_val)
            y_pred_classes = np.where(y_pred > 0, 1, -1)
            accuracy = accuracy_score(y_val, y_pred_classes)
        score = model.obj(X_val, y_val, loss=model.loss)[0] / len(y_val)
        return {'params': param_dict, 'score': score, 'accuracy':accuracy}
    
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_single_params)(params) for params in param_combinations
    )
    

    best_result = min(results, key=lambda x: x['score'])
    
    return best_result['params'], best_result['score'], best_result['accuracy'], results



## Parameters to be selected
param_grid = {
    'constraint': [[], [{'name': '>=0'}]],
    'biased': [True, False],
    'rank': [5, 10],
    'C': [0.0006, 0.0002],
    'rho': [0.3, 0.6],
}




print("----------------------------Part 1: Regression Task----------------------------")
## Choose Absolute Loss
fixed_params = {
    'loss': {'name': 'mae'},       
    'n_users': user_num,            
    'n_items': item_num,            
    'max_iter': 100000,
    'max_iter_CD': 5, 
    'init_mean': 3.0,
    'verbose': verbose
}

best_params, best_score, _, all_results = parallel_grid_search(plqMF_Ridge, param_grid, fixed_params, X_train, y_train, X_test, y_test, n_jobs)
print("\n" + "="*50)
print("BEST RESULTS(Using Absolute Loss)")
print("="*50)
print("Optimal Parameters:")
for param, value in best_params.items():
    print(f"  {param:12}: {value}")
print(f"\nBest Validation MAE: {best_score:.4f}")
print("="*50)


## Choose Square Loss
fixed_params = {
    'loss': {'name': 'MSE'},       
    'n_users': user_num,            
    'n_items': item_num,            
    'max_iter': 100000,
    'max_iter_CD': 5,
    'init_mean': 3.0,
    'verbose': verbose
}

best_params, best_score, _, all_results = parallel_grid_search(plqMF_Ridge, param_grid, fixed_params, X_train, y_train, X_test, y_test, n_jobs)
print("\n" + "="*50)
print("BEST RESULTS(Using Square Loss)")
print("="*50)
print("Optimal Parameters:")
for param, value in best_params.items():
    print(f"  {param:12}: {value}")
print(f"\nBest Validation MSE: {best_score:.4f}")
print("="*50)



print("\n\n----------------------------Part 2: Classification Task----------------------------")
##  Classification Label Data
y_train_bin = np.where(y_train > 3.0, 1, -1)
y_test_bin = np.where(y_test > 3.0, 1, -1)


## Choose Hinge Loss
fixed_params = {
    'loss': {'name': 'hinge'},       
    'n_users': user_num,            
    'n_items': item_num,            
    'max_iter': 100000,
    'max_iter_CD': 5,
    'init_sd': 1,
    'verbose': verbose
}

best_params, best_score, best_acc, all_results = parallel_grid_search(plqMF_Ridge, param_grid, fixed_params, X_train, y_train_bin, X_test, y_test_bin, n_jobs)
print("\n" + "="*50)
print("BEST RESULTS(Using Hinge Loss)")
print("="*50)
print("Optimal Parameters:")
for param, value in best_params.items():
    print(f"  {param:12}: {value}")
print(f"\nBest Validation Accuracy: {best_acc:.4f}")
print("="*50)


## Choose Hinge Square Loss
fixed_params = {
    'loss': {'name': 'svm square'},       
    'n_users': user_num,            
    'n_items': item_num,            
    'max_iter': 100000,
    'max_iter_CD': 5,
    'init_sd': 1,
    'verbose': verbose
}

best_params, best_score, best_acc, all_results = parallel_grid_search(plqMF_Ridge, param_grid, fixed_params, X_train, y_train_bin, X_test, y_test_bin, n_jobs)
print("\n" + "="*50)
print("BEST RESULTS(Using Hinge Square Loss)")
print("="*50)
print("Optimal Parameters:")
for param, value in best_params.items():
    print(f"  {param:12}: {value}")
print(f"\nBest Validation Accuracy: {best_acc:.4f}")

print("="*50)
