### ridge_quantile

| metric | california_housing | make_friedman1_5k_100 | make_regression_100k |
| --- | --- | --- | --- |
| n_samples | 20640 | 5000 | 100000 |
| n_features | 8 | 100 | 20 |
| n_candidates | 6 | 6 | 6 |
| elapsed_sec_mean | 6.682 | 4.133 | 3.013 |
| elapsed_sec_std | 0.7097 | 0.008051 | 0.01155 |
| best_C | 0.001 | 10 | 10 |
| best_mse | 1.21 | 9.622 | 146.6 |

### ridge_quantile_monotonic

| metric | california_housing | make_friedman1_5k_100 | make_regression_100k |
| --- | --- | --- | --- |
| n_samples | 20640 | 5000 | 100000 |
| n_features | 8 | 100 | 20 |
| n_candidates | 6 | 6 | 6 |
| elapsed_sec_mean | 51.22 | 47.93 | 44.69 |
| elapsed_sec_std | 0.011 | 0.4864 | 0.1447 |
| best_C | 10 | 1 | 10 |
| best_mse | 2.034 | 37.46 | 2.997e+04 |

### ridge_quantile_eps

| metric | california_housing | make_regression_100k |
| --- | --- | --- |
| n_samples | 20640 | 100000 |
| n_features | 8 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 5.95 | 4.136 |
| elapsed_sec_std | 0.02343 | 0.01799 |
| best_C | 0.001 | 10 |
| best_mse | 1.278 | 146.6 |

### ridge_mae

| metric | california_housing | make_regression_100k |
| --- | --- | --- |
| n_samples | 20640 | 100000 |
| n_features | 8 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 5.96 | 3.049 |
| elapsed_sec_std | 0.01187 | 0.02813 |
| best_C | 0.001 | 1 |
| best_mse | 0.8453 | 100.6 |

### ridge_huber

| metric | california_housing | make_regression_100k |
| --- | --- | --- |
| n_samples | 20640 | 100000 |
| n_features | 8 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 5.822 | 1.738 |
| elapsed_sec_std | 0.01893 | 0.006578 |
| best_C | 0.001 | 1 |
| best_mse | 0.7312 | 100.6 |

### ridge_svr

| metric | california_housing | make_regression_100k |
| --- | --- | --- |
| n_samples | 20640 | 100000 |
| n_features | 8 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 6.655 | 2.153 |
| elapsed_sec_std | 0.01025 | 0.009999 |
| best_C | 0.001 | 1 |
| best_mse | 0.8383 | 100.6 |

### elasticnet_quantile

| metric | california_housing | make_friedman1_5k_100 | make_regression_100k |
| --- | --- | --- | --- |
| n_samples | 20640 | 5000 | 100000 |
| n_features | 8 | 100 | 20 |
| n_candidates | 6 | 6 | 6 |
| elapsed_sec_mean | 7.867 | 6.285 | 4.448 |
| elapsed_sec_std | 0.008464 | 0.01016 | 0.01604 |
| best_C | 0.001 | 10 | 10 |
| best_mse | 1.269 | 9.616 | 146.6 |

### elasticnet_quantile_monotonic

| metric | california_housing | make_friedman1_5k_100 | make_regression_100k |
| --- | --- | --- | --- |
| n_samples | 20640 | 5000 | 100000 |
| n_features | 8 | 100 | 20 |
| n_candidates | 6 | 6 | 6 |
| elapsed_sec_mean | 59.59 | 62.67 | 53.01 |
| elapsed_sec_std | 0.0171 | 0.408 | 0.05259 |
| best_C | 10 | 10 | 10 |
| best_mse | 2.034 | 32.41 | 2.997e+04 |

### elasticnet_quantile_eps

| metric | california_housing | make_regression_100k |
| --- | --- | --- |
| n_samples | 20640 | 100000 |
| n_features | 8 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 7.717 | 4.012 |
| elapsed_sec_std | 0.004766 | 0.01047 |
| best_C | 0.01 | 10 |
| best_mse | 1.219 | 146.6 |

### elasticnet_mae

| metric | california_housing | make_regression_100k |
| --- | --- | --- |
| n_samples | 20640 | 100000 |
| n_features | 8 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 7.365 | 2.41 |
| elapsed_sec_std | 0.01046 | 0.0114 |
| best_C | 0.001 | 1 |
| best_mse | 0.726 | 100.6 |

### elasticnet_huber

| metric | california_housing | make_regression_100k |
| --- | --- | --- |
| n_samples | 20640 | 100000 |
| n_features | 8 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 7.141 | 2.067 |
| elapsed_sec_std | 0.01124 | 0.00853 |
| best_C | 0.001 | 1 |
| best_mse | 0.6835 | 100.6 |

### elasticnet_svr

| metric | california_housing | make_regression_100k |
| --- | --- | --- |
| n_samples | 20640 | 100000 |
| n_features | 8 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 8.203 | 2.448 |
| elapsed_sec_std | 0.007247 | 0.003593 |
| best_C | 0.001 | 1 |
| best_mse | 0.7173 | 100.6 |

### ridge_svm

| metric | digits_low_high | openml_bioresponse | make_classification_100k |
| --- | --- | --- | --- |
| n_samples | 1797 | 3751 | 100000 |
| n_features | 64 | 1776 | 20 |
| n_candidates | 6 | 6 | 6 |
| elapsed_sec_mean | 0.6506 | 42.35 | 3.048 |
| elapsed_sec_std | 0.006257 | 0.1496 | 0.01226 |
| best_C | 0.1 | 0.001 | 0.01 |
| best_accuracy | 0.8826 | 0.7713 | 0.9295 |

### ridge_smooth_svm

| metric | digits_low_high | make_classification_100k |
| --- | --- | --- |
| n_samples | 1797 | 100000 |
| n_features | 64 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 0.2576 | 3.737 |
| elapsed_sec_std | 0.003365 | 0.0104 |
| best_C | 0.01 | 0.01 |
| best_accuracy | 0.8804 | 0.9294 |

### ridge_squared_svm

| metric | digits_low_high | make_classification_100k |
| --- | --- | --- |
| n_samples | 1797 | 100000 |
| n_features | 64 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 0.8605 | 21.28 |
| elapsed_sec_std | 0.003475 | 0.06886 |
| best_C | 0.01 | 0.001 |
| best_accuracy | 0.8781 | 0.9293 |

### elasticnet_svm

| metric | digits_low_high | openml_bioresponse | make_classification_100k |
| --- | --- | --- | --- |
| n_samples | 1797 | 3751 | 100000 |
| n_features | 64 | 1776 | 20 |
| n_candidates | 6 | 6 | 6 |
| elapsed_sec_mean | 0.9804 | 71.56 | 3.76 |
| elapsed_sec_std | 0.009102 | 0.1047 | 0.01082 |
| best_C | 0.1 | 0.01 | 0.01 |
| best_accuracy | 0.8815 | 0.7726 | 0.9295 |

### elasticnet_smooth_svm

| metric | digits_low_high | make_classification_100k |
| --- | --- | --- |
| n_samples | 1797 | 100000 |
| n_features | 64 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 0.3261 | 4.556 |
| elapsed_sec_std | 0.005569 | 0.01597 |
| best_C | 0.1 | 0.01 |
| best_accuracy | 0.8776 | 0.9295 |

### elasticnet_squared_svm

| metric | digits_low_high | make_classification_100k |
| --- | --- | --- |
| n_samples | 1797 | 100000 |
| n_features | 64 | 20 |
| n_candidates | 6 | 6 |
| elapsed_sec_mean | 1.085 | 25.58 |
| elapsed_sec_std | 0.004304 | 0.07582 |
| best_C | 0.01 | 0.001 |
| best_accuracy | 0.877 | 0.9294 |

## Config

```json
{
  "C_grid": [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0
  ],
  "cv": 3,
  "l1_ratio_grid": [
    0.2
  ],
  "max_iter": 5000000,
  "n_jobs": 3,
  "output_dir": "benchmarks/results",
  "preprocess_X": "standard",
  "quantile_grid": [
    0.25
  ],
  "repeats": 3,
  "task_datasets": {
    "elasticnet_huber": [
      "california_housing",
      "make_regression_100k"
    ],
    "elasticnet_mae": [
      "california_housing",
      "make_regression_100k"
    ],
    "elasticnet_quantile": [
      "california_housing",
      "make_friedman1_5k_100",
      "make_regression_100k"
    ],
    "elasticnet_quantile_eps": [
      "california_housing",
      "make_regression_100k"
    ],
    "elasticnet_quantile_monotonic": [
      "california_housing",
      "make_friedman1_5k_100",
      "make_regression_100k"
    ],
    "elasticnet_smooth_svm": [
      "digits_low_high",
      "make_classification_100k"
    ],
    "elasticnet_squared_svm": [
      "digits_low_high",
      "make_classification_100k"
    ],
    "elasticnet_svm": [
      "digits_low_high",
      "openml_bioresponse",
      "make_classification_100k"
    ],
    "elasticnet_svr": [
      "california_housing",
      "make_regression_100k"
    ],
    "ridge_huber": [
      "california_housing",
      "make_regression_100k"
    ],
    "ridge_mae": [
      "california_housing",
      "make_regression_100k"
    ],
    "ridge_quantile": [
      "california_housing",
      "make_friedman1_5k_100",
      "make_regression_100k"
    ],
    "ridge_quantile_eps": [
      "california_housing",
      "make_regression_100k"
    ],
    "ridge_quantile_monotonic": [
      "california_housing",
      "make_friedman1_5k_100",
      "make_regression_100k"
    ],
    "ridge_smooth_svm": [
      "digits_low_high",
      "make_classification_100k"
    ],
    "ridge_squared_svm": [
      "digits_low_high",
      "make_classification_100k"
    ],
    "ridge_svm": [
      "digits_low_high",
      "openml_bioresponse",
      "make_classification_100k"
    ],
    "ridge_svr": [
      "california_housing",
      "make_regression_100k"
    ]
  },
  "tol": 0.0001
}
```
