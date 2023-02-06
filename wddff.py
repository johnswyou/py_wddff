from dwt import DwtTransformer, LagTransformer
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from utils import train_val_test_split, normalize_train_val_test_X
from sklearn.feature_selection import mutual_info_regression
from itertools import compress

def single_config_wddff(X, y, leadtime, wavelet = None, j = None, max_wavelet_length = 14, max_j = 6, atrous = False, lag_X = 14, lag_Y = 14, normalize = True, mutual_info_thresh = None, keep_target_inputs = True):

    # X and y should have the same index!!!

    if not isinstance(X, pd.DataFrame):
        raise ValueError(f'X must be of type pd.DataFrame, not {type(X)}')

    if not isinstance(y, pd.DataFrame):
        raise ValueError(f'y must be of type pd.DataFrame, not {type(y)}')

    target_name = list(y)[0] # save the target column name

    if target_name in list(X):
        X = X.drop(target_name, axis=1)

    # MODWT and lag

    if wavelet is None or j is None:

        X_pipeline = make_pipeline(LagTransformer(lag_X))
        Y_pipeline = make_pipeline(LagTransformer(lag_Y))

    else:
        X_pipeline = make_pipeline(DwtTransformer(wavelet, j, max_wavelet_length, max_j),
                                LagTransformer(lag_X))

        Y_pipeline = make_pipeline(DwtTransformer(wavelet, j, max_wavelet_length, max_j),
                                LagTransformer(lag_Y))

    X = X_pipeline.fit_transform(X)
    y = Y_pipeline.fit_transform(y)

    assert X.shape[0] == y.shape[0]
    assert target_name != 'target'

    # Craete the actual target

    if keep_target_inputs:
        y['target'] = y[target_name].shift(-leadtime)
        full_set = pd.concat([y, X], axis=1)
    else:
        assert 'target' not in list(X)
        X['target'] = y[target_name].shift(-leadtime)
        full_set = X
        
    full_set.dropna(inplace=True)

    # Train, Validation, Test Split
    # We use 70%, 15% and 15% for trian, validation and test sets, respectively

    train_set, val_set, test_set = train_val_test_split(full_set)

    X_train = train_set.drop('target', axis=1)
    y_train = train_set[['target']]

    X_val = val_set.drop('target', axis=1)
    y_val = val_set[['target']]

    X_test = test_set.drop('target', axis=1)
    y_test = test_set[['target']]

    # Normalize
    if normalize:
        X_train, X_val, X_test = normalize_train_val_test_X(X_train, X_val, X_test)

    # *************************************************************************************

    # mutual_info_thresh is an experimental feature; likely will be removed in the future

    X_colnames = list(X_train)

    # Mutual information input variable selection based on a threshold
    if mutual_info_thresh is not None:
        assert (mutual_info_thresh < 1) and (mutual_info_thresh > 0)
        scores = mutual_info_regression(X_train.to_numpy(), y_train.to_numpy().squeeze())
        boolean_mask = [score > mutual_info_thresh for score in scores.tolist()]
        X_train = X_train.loc[:, boolean_mask]
        X_val = X_val.loc[:, boolean_mask]
        X_test = X_test.loc[:, boolean_mask]
        # print(f'Input features: {list(compress(X_colnames, boolean_mask))}')
        print(list(X_train))
    # else:
    #     print(f'Input features: {X_colnames}')

    # *************************************************************************************

    return X_train, X_val, X_test, y_train, y_val, y_test

def multi_config_wddff(X, y, leadtime, wavelet = None, j = None, max_wavelet_length = 14, max_j = 6, atrous = False, lag_X = 14, lag_Y = 14, normalize = True, keep_target_inputs = True):

    # wavelet (list) should take form [y_wavelet, x1_decomp_level, x2_decomp_level, ...]
    # j (list) should take form [y_decomp_level, x1_decomp_level, x2_decomp_level, ...]
    # when keep_target_inputs is False, wavelet and j should be lists with length ncol(X)

    if not isinstance(X, pd.DataFrame):
        raise ValueError(f'X must be of type pd.DataFrame, not {type(X)}')

    if not isinstance(y, pd.DataFrame):
        raise ValueError(f'y must be of type pd.DataFrame, not {type(y)}')

    # save the target column name

    target_name = list(y)[0]

    # If target is in X, remove it from X

    if target_name in list(X):
        X = X.drop(target_name, axis=1)

    # **************
    # MODWT and lag
    # **************

    if wavelet is None or j is None:

        X_pipeline = make_pipeline(LagTransformer(lag_X))
        Y_pipeline = make_pipeline(LagTransformer(lag_Y))

        X = X_pipeline.fit_transform(X)
        y = Y_pipeline.fit_transform(y)

    else:

        assert isinstance(wavelet, list)
        assert isinstance(j, list)

        if keep_target_inputs:
            assert len(list(X)) + 1 == len(wavelet) == len(j)
        else:
            assert len(list(X)) == len(wavelet) == len(j)

            # Does not matter what the first element of wavelet and j are.
            # TO DO: Change this implementation of dealing with keep_target_inputs=False;
            #        at present, we are doing the unecessary wavelet decomposition of y.
            wavelet = ['haar'] + wavelet # haar chosen arbitrarily; does not matter which wavelet
            j = [1] + j                  # 1 chosen arbitrarily does not matter which j

        # y

        Y_pipeline = make_pipeline(DwtTransformer(wavelet[0], j[0], max_wavelet_length, max_j),
                                   LagTransformer(lag_Y))

        y = Y_pipeline.fit_transform(y)

        # X

        X_storage = []

        for i in range(1, len(wavelet)):

            X_pipeline = make_pipeline(DwtTransformer(wavelet[i], j[i], max_wavelet_length, max_j, keep_original=True),
                                       LagTransformer(lag_X))

            X_storage.append(X_pipeline.fit_transform(X.iloc[:, i-1].to_frame()))

        X = pd.concat(X_storage, axis=1)

    assert X.shape[0] == y.shape[0]
    assert target_name != 'target'

    # Craete the actual target

    y['target'] = y[target_name].shift(-leadtime)

    if keep_target_inputs:
        full_set = pd.concat([y, X], axis=1)
    else:
        full_set = pd.concat([y[['target']], X], axis=1)

    full_set.dropna(inplace=True)

    # ******************************
    # Train, Validation, Test Split
    # ******************************

    # We use 70%, 15% and 15% for trian, validation and test sets, respectively

    train_set, val_set, test_set = train_val_test_split(full_set)

    X_train = train_set.drop('target', axis=1)
    y_train = train_set[['target']]

    X_val = val_set.drop('target', axis=1)
    y_val = val_set[['target']]

    X_test = test_set.drop('target', axis=1)
    y_test = test_set[['target']]

    # Normalize
    if normalize:
        X_train, X_val, X_test = normalize_train_val_test_X(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test



    



