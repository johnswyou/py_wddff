from filters import scaling_filter
import numpy as np
from sklearn.utils.validation import check_array, check_X_y
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import pandas as pd
import itertools

class Dwt:

    def __init__(self, X, wavelet, j):
        
        if type(X) is np.ndarray:
            self.X = X.tolist()
        elif isinstance(X, list):
            self.X = X
        else:
            raise ValueError

        # Add error catch for upper limit on decomposition level based on length of
        # wavelet filter chosen and number of observations in data set

        self.wavelet = wavelet
        self.j = j
        self.N = len(X)                       # need to test for list, numpy array and pandas series
        self.g = scaling_filter(self.wavelet) # a list
        self.L = len(self.g)
        self.h = self.wavelet_filter()
        self.n_boundary_coefs = ((2**self.j) - 1) * (self.L - 1) + 1 # See Eq. 10 in Basta [2014]
        
    def coefs(self, type, x=None, j=None):

        if type == "scaling":
            filtr = self.g
        elif type == "wavelet":
            filtr = self.h
        else:
            raise ValueError('For the type argument, use one of the following: scaling, wavelet')

        if x is None:
            x = self.X
        elif isinstance(x, list):
            pass
        else:
            raise TypeError

        if j is None:
            j = self.j
        elif isinstance(j, int):
            pass
        else:
            raise TypeError

        # x MUST be a list

        N = len(x)

        c = np.zeros((N, ))
        # c = [0]*N

        for k in range(N, 2**(j+1), -1):

            tmp = 0 # intialize sum

            for i in range(1, self.L+1):

                pos = k - (2**(j-1))*(i-1) # time position
                if pos <= 0:
                    pos = N + pos # circularity assumption
                tmp = tmp + filtr[i-1] * x[pos-1]

            c[k-1] = tmp # scaling coefficient at time position k

        return c

    def wavelet_filter(self):
        
        h = [0]*self.L
        for i in range(0, self.L):
            h[self.L-i-1] = self.g[i] / ((-1)**(i+1))
        
        return h

    def modwt(self, atrous=False):

        x = 2*self.X # self.X is a list
        v = np.zeros((2*self.N, self.j))
        w = np.zeros((2*self.N, self.j))

        for j in range(1, self.j+1):

            # calculate scaling and wavelet coefficients
            v[:,j-1] = self.coefs(type="scaling", x=x, j=j) # Eq. 14 in Basta [2014]

            if atrous:
                w[:,j-1] = np.array(x) - v[:,j-1] # see Eq. 6 in Maheswaran and Khosa [2012]
            else:
                w[:,j-1] = self.coefs(type="wavelet", x=x, j=j) # Eq. 13 in Basta [2014]

            x = v[:,j-1].tolist()

        W = np.column_stack((w[range(self.N, 2*self.N),:], v[range(self.N, 2*self.N),self.j-1]))

        return W

class DwtTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet, j, max_wavelet_length, max_j, atrous = False, remove_boundary_coefs = True): # no *args or **kargs

        self.wavelet = wavelet
        self.j = j
        self.max_wavelet_length = max_wavelet_length
        self.max_j = max_j
        self.atrous = atrous
        self.remove_boundary_coefs = remove_boundary_coefs

    def fit(self, X, y=None): # y is ignored

        # Input validation

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f'X must be of type pd.DataFrame, not {type(X)}')

        self.X_colnames = list(X) # list containing strings that denote each column in DataFrame X
        self.n_boundary_coefs = ((2**self.max_j) - 1) * (self.max_wavelet_length - 1) + 1 # See Eq. 10 in Basta [2014]

        return self

    def transform(self, X, y=None): # y is ignored

        # I assume .transform() is called immediately after .fit()

        # The following: 
        # 1. performs MODWT on every column of X and 
        # 2. concatenates the new wavelet features to the original X.

        storage = [X]

        for i in range(len(self.X_colnames)):
            x = X.iloc[:, i].tolist()                                                                     # take the ith column (time series) in X and convert to a list
            dwt_instance = Dwt(x, self.wavelet, self.j)
            x_post_dwt = dwt_instance.modwt(atrous=self.atrous)                                           # Perform MODWT on ith column of X
            ncols_x_post_dwt = x_post_dwt.shape[1]                                                        # This will always be self.j + 1
            x_colnames = ["W"+str(k) for k in range(1, ncols_x_post_dwt)] + ["V" + str(ncols_x_post_dwt)] # Make column names
            x_colnames = [self.X_colnames[i] + '_' + elem for elem in x_colnames]                         # Make column names
            df = pd.DataFrame(x_post_dwt, columns = x_colnames)                                           # Convert ndarray to pandas data frame
            storage.append(df)

        res = pd.concat(storage, axis=1)

        # Remove boundary coefficients

        if self.remove_boundary_coefs:
            res = res.iloc[self.n_boundary_coefs:]
        
        return res

class LagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lag_X):

        self.lag_X = lag_X

    def fit(self, X, y=None): # y is ignored
        
        # Input validation

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f'X must be of type pd.DataFrame, not {type(X)}')

        self.X_colnames = list(X)

        return self

    def transform(self, X, y = None): # y is ignored

        # I assume .transform() is called immediately after .fit()

        new_X_colnames = [column_name + f'_lag{lag}' for column_name in self.X_colnames for lag in range(1, self.lag_X+1)]
        new_X_columns = [X.loc[:, column_name].shift(lag) for column_name in self.X_colnames for lag in range(1, self.lag_X+1)]
        df = pd.concat(new_X_columns, axis=1)
        df.columns = new_X_colnames

        lagged_X = pd.concat([X, df], axis=1)
        lagged_X = lagged_X.iloc[self.lag_X:] # Remove NaN values
        self.lagged_X = lagged_X

        return self.lagged_X

class SovEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, cutoff_0=True):
        
        self.order = 2
        self.cutoff_0 = cutoff_0
    
    def fit(self, X, y):

        X, y = check_X_y(X, y)
        self.X = X
        self.y = y

        # X = check_array(X)
        
        n_cols = X.shape[1]
        combns = list(itertools.combinations(range(1,n_cols+self.order), self.order))
        combns = [(i, j-self.order+1) for i, j in combns]

        res = np.zeros((X.shape[0], len(combns)))

        for i in range(len(combns)):
            current_combn = combns[i]
            current_col_1, current_col_2 = current_combn
            current_col_1 = current_col_1 - 1
            current_col_2 = current_col_2 - 1
            temp = np.multiply(X[:, current_col_1], X[:, current_col_2])
            res[:, i] = temp

        Xt = np.hstack((X, res))
        self.Xtc = np.column_stack((np.ones(X.shape[0]), Xt))
        
        self.H = np.matmul(np.linalg.pinv(self.Xtc), y)

        return self

    def predict(self, X):

        # check_is_fitted(self)
        X = check_array(X)

        n_cols = X.shape[1]
        combns = list(itertools.combinations(range(1,n_cols+self.order), self.order))
        combns = [(i, j-self.order+1) for i, j in combns]

        res = np.zeros((X.shape[0], len(combns)))

        for i in range(len(combns)):
            current_combn = combns[i]
            current_col_1, current_col_2 = current_combn
            current_col_1 = current_col_1 - 1
            current_col_2 = current_col_2 - 1
            temp = np.multiply(X[:, current_col_1], X[:, current_col_2])
            res[:, i] = temp

        Xt = np.hstack((X, res))
        Xtc = np.column_stack((np.ones(X.shape[0]), Xt))

        res = np.matmul(Xtc, self.H)

        if self.cutoff_0:
            res = res.clip(min=0)

        return res
        
