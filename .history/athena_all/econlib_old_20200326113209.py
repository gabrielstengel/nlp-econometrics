import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import *
import statsmodels.api as sm
from linearmodels import PanelOLS
from linearmodels.iv import *
from statsmodels.tsa.api import VAR
import statsmodels.tsa.ar_model as s_ar
import statsmodels.tsa.stattools as s_st
from statsmodels.multivariate.pca import PCA
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


# argument: vector x
# returns mean of x
def findMean(x):
    return np.mean(x)


# argument: vector x
# returns standard devation of x
def findStd(x):
    return np.std(x)


# argument: vector x
# returns variance of x
def findVar(x):
    return np.var(x)


# argument: vector x
# returns maximum value of x
def findMax(x):
    return np.max(x)


# argument: vector x
# returns minimum value of x
def findMin(x):
    return np.min(x)


# argument: vector x
# returns median value of x
def findMedian(x):
    return np.median(x)

   