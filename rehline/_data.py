import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler



def make_fair_classification(n_samples=100, n_features=5, ind_sensitive=0):
    """
    Generate a random binary fair classification problem.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=5
        The total number of features. 

    ind_sensitive : int, default=0
        The index of the sensitive feature.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The +/- labels for class membership of each sample.

    X_sen: ndarray of shape (n_samples,)
        The centered samples of the sensitive feature.
    """

    X, y = make_classification(n_samples, n_features)
    y = 2*y - 1

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_sen = X[:, ind_sensitive]

    return X, y, X_sen



def load_dataset(name, return_X_y=False):
    """
    Load a CSV dataset from the rehline/datasets directory.

    This function follows scikit-learn's dataset loading conventions, providing
    flexible return formats and metadata.

    Parameters
    ----------
    name : str
        Name of the dataset (without the .csv extension, e.g., "ml-100k").
    return_X_y : bool, default=False
        If True, returns a tuple (X, y) instead of a dictionary.

    Returns
    -------
    data : dict
        When return_X_y=False, a dictionary containing:
        - data: Feature matrix X
        - target: Target vector y
        - feature_names: List of feature column names
        - target_names: List containing the target column name
        - DESCR: Dataset description
    X, y : tuple of ndarrays
        When return_X_y=True, returns the feature matrix and target vector.

    Raises
    ------
    FileNotFoundError
        If the specified dataset file does not exist in rehline/datasets.
    """
    # Determine dataset path
    data_dir = Path(__file__).parent / "datasets"
    file_path = data_dir / f"{name}.csv"

    # Validate file existence
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file {file_path} not found. Please check the name or ensure "
            "the file is placed in the rehline/datasets directory."
        )

    # Load CSV (assumes last column is target, others are features)
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values  # Feature matrix
    y = df.iloc[:, -1].values   # Target vector (last column)

    # Prepare metadata
    feature_names = df.columns[:-1].tolist()
    target_name = df.columns[-1] if len(df.columns) > 0 else "target"
    
    # Construct return dictionary
    data_dict = {
        "data": X,
        "target": y,
        "feature_names": feature_names,
        "target_names": [target_name],
        "DESCR": f"Dataset {name} loaded from rehline/datasets/{name}.csv"
    }

    # Return appropriate format based on flag
    if return_X_y:
        return X, y
    else:
        return data_dict
