from filters import scaling_filter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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
    def __init__(self, wavelet, j, max_wavelet_length, max_j, atrous = False): # no *args or **kargs
        
        self.wavelet = wavelet
        self.j = j
        self.max_wavelet_length = max_wavelet_length
        self.max_j = max_j
        self.atrous = atrous
        self.n_boundary_coefs = ((2**self.max_j) - 1) * (self.max_wavelet_length - 1) + 1 # See Eq. 10 in Basta [2014]

    def fit(self, X):
        assert isinstance(X, pd.DataFrame)
        return self # do nothing

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        storage = [X]
        
        X_colnames = list(X.columns)

        for i in range(len(X.columns)):
            x = X.iloc[:, i].tolist()
            dwt_instance = Dwt(x, self.wavelet, self.j)
            x_post_dwt = dwt_instance.modwt(atrous=self.atrous)
            ncols_x_post_dwt = x_post_dwt.shape[1]
            x_colnames = ["v"+str(k) for k in range(1, ncols_x_post_dwt)] + ["w" + str(ncols_x_post_dwt)]
            x_colnames = [X_colnames[i] + '_' + elem for elem in x_colnames]
            df = pd.DataFrame(x_post_dwt, columns = x_colnames)
            storage.append(df)

        res = pd.concat(storage, axis=1)
        res = res.iloc[self.n_boundary_coefs:] # Remove boundary coefficients
        
        return res
