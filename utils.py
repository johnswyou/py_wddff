import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.feature_selection import mutual_info_regression

# Reference: https://stackoverflow.com/questions/18495098/python-check-if-an-object-is-a-list-of-strings
def is_list_of_strings(lst):
    if lst and isinstance(lst, list):
        return all(isinstance(elem, str) for elem in lst)
    else:
        return False

def train_val_test_split(X, train_split=0.7, val_split=0.15, test_split = 0.15):

    assert train_split + val_split + test_split == 1.

    if not isinstance(X, pd.DataFrame):
        raise ValueError(f'X must be of type pd.DataFrame, not {type(X)}')

    n_rows_X = len(X)
    
    num_train_samples = int(train_split * n_rows_X)
    num_val_samples = int(val_split * n_rows_X)
    # num_test_samples = n_rows_X - (num_train_samples + num_val_samples)

    train, val, test = np.split(X, [num_train_samples, num_train_samples + num_val_samples])

    return train, val, test

def ts_train_val_test(X, y, lead_time, look_back, normalize = False, mutual_info_thresh = None, train_split = 0.7, val_split = 0.15, test_split = 0.15, batch_size = 32, sampling_rate = 1):
    
    assert train_split+val_split+test_split==1

    # Make sure both X and y are pandas data frames

    if not isinstance(X, pd.DataFrame):
        raise ValueError(f'X must be of type pd.DataFrame, not {type(X)}')

    if not isinstance(y, pd.DataFrame):
        raise ValueError(f'y must be of type pd.DataFrame, not {type(y)}')

    # Convert X and y to numpy arrays

    X = X.to_numpy()
    y = y.to_numpy().squeeze()

    sequence_length = look_back + 1
    delay = sampling_rate * (sequence_length + lead_time - 1)

    num_train_samples = int(train_split * len(X))
    num_val_samples = int(val_split * len(X))
    # num_test_samples = len(X) - num_train_samples - num_val_samples

    # Normalize (if normalize = True)
    if normalize:
        mean = X[:num_train_samples].mean(axis=0)
        X -= mean
        std = X[:num_train_samples].std(axis=0)
        X /= std

    # Mutual information input variable selection based on a threshold
    if mutual_info_thresh is not None:
        scores = mutual_info_regression(X.to_numpy(), y.to_numpy().squeeze())
        boolean_mask = [score > mutual_info_thresh for score in scores.tolist()]
        X = X.loc[:, boolean_mask]

    train_dataset = keras.utils.timeseries_dataset_from_array(
        X[:-delay],
        targets=y[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=0,
        end_index=num_train_samples)
    
    val_dataset = keras.utils.timeseries_dataset_from_array(
        X[:-delay],
        targets=y[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=num_train_samples,
        end_index=num_train_samples + num_val_samples)
    
    test_dataset = keras.utils.timeseries_dataset_from_array(
        X[:-delay],
        targets=y[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=num_train_samples + num_val_samples)

    return train_dataset, val_dataset, test_dataset

def nse(obs, pred):

    assert isinstance(obs, np.array)
    assert isinstance(pred, np.array)

    assert obs.shape[1] != 1
    assert pred.shape[1] != 1

    numerator = np.sum(np.square(obs - pred))
    denominator = np.sum(np.square(obs - np.mean(obs)))

    nash = 1 - (numerator/denominator)

    return nash[0]

