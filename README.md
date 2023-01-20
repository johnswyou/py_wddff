# py_wddff

Wavelet data driven forecasting framework (WDDFF) implementation in Python 3

## Description

This is the development repository for a Python 3 module implementing WDDFF first introduced by Quilty and Adamowski in:

Quilty, J., &amp; Adamowski, J. (2018). Addressing the incorrect usage of wavelet-based hydrological and water resources forecasting models for real-world applications with best practices and a new forecasting framework. Journal of Hydrology, 563, 336–353. https://doi.org/10.1016/j.jhydrol.2018.05.003 

## Prerequisites

For input variable selection functions such as `pcis_bic`, the `rpy2` package was leveraged because our
input variable selection methods were originally implemented in the R programming language. The user must
have a local R installation and furthermore must have the `hydroIVS` package installed which can be installed
as follows:

```r
# install.packages("devtools")
devtools::install_github("johnswyou/hydroIVS")
```

## Discrete Wavelet Transforms

The `Dwt` class implements both Maximal Overlap Discrete Wavelet Transform (MODWT) and Atrous algorithm.

```python
import numpy as np
from dwt import Dwt

rg = np.random.default_rng(1426)
N = 1000
X = rg.random((N,))

dwt_instance = Dwt(X, "haar", 3)

# To compute MODWT
dwt_instance.modwt(atrous=False)

# To compute Atrous
dwt_instance.modwt(atrous=True)
```
