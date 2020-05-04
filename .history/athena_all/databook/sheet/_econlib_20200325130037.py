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

from athena_all.file_processing.excel import DataReader



# argument: vector x
# returns mean of x
def findMean(self, x):
    return np.mean(x)


# argument: vector x
# returns standard devation of x
def findStd(self, x):
    return np.std(x)


# argument: vector x
# returns variance of x
def findVar(self, x):
    return np.var(x)


# argument: vector x
# returns maximum value of x
def findMax(self, x):
    return np.max(x)


# argument: vector x
# returns minimum value of x
def findMin(self, x):
    return np.min(x)


# argument: vector x
# returns median value of x
def findMedian(self, x):
    return np.median(x)

# arguments: s = Sheet. yColName and xColName = column names.
# returns: their correlation
def findCorr(self, yColName, xColName):
    return self.corr_matrix[yColName][findColumnIndexGivenName(self.df, xColName)]


# arguments: s = Sheet. col = name of column.
# returns: name of column in s's df most correlated with col.
def largestCorr(self, col):

    m = self.corr_matrix
    col_corrs = m[[col]]
    champ = -1
    max = -2

    for i in range(0, len(col_corrs)):
        
        c = abs(col_corrs.iloc[i, 0])
        if c > max and c < 1:
            max = c
            champ = i

    if (champ == -1):
        print("ERROR: none found")
        return
    
    return findVariableName(m, champ)


# arguments: s = Sheet. col = name of column. num_return = number seeked to return.
# returns: names of num_return columns in s's df most correlated with col.
def largestCorrList(self, col, num_return=3):
    
    df = self.df
    m = self.corr_matrix
    v = abs(m[[col]])
    n = len(v)
    vAbs = np.zeros(n)

    for i in range(0, n):
        vAbs[i] = v.iloc[i,0]

    # we want the (num_return + 1)'th largest because we don't it to count itself
    p = _kthLargest(vAbs, num_return + 1)

    returnVector = []

    for i in range(0, n):
        c = vAbs[i]
        if c >= p and c < 1:
            returnVector.append(findVariableName(m, i))

    return returnVector


# arguments: s = Sheet. num_return = number seeked for return.
# returns: largest num_return pairwise correlations overall in the dataset (string column names)
def overallLargestCorrs(self, num_return=5):
    
    m = self.corr_matrix
    n = len(m)

    v = np.zeros(n*n)

    for i in range(0, n):
        for j in range(i, n):
            element = abs(m.iloc[i,j])
            v[_dfToVectorIndex(n, i, j)] = element
    
    p = _kthLargest(v, num_return + 1)
    r = np.zeros(num_return)
    j = 0

    for i in range(0, len(v)):
        vi = v[i]
        if vi >= p and vi < 1:
            r[j] = i
            j += 1

    s = []

    for i in range(0, len(r)):
        ri = r[i]
        t = _vectortoDFIndex(n, ri)
        s.append(findVariableName(m, t))

    return s

# returns 1d index of (x, y). we traverse down the first column and then down the second column etc
def _dfToVectorIndex(numRows, x, y):
    return (x * numRows) + y

# returns 2d index of x. we traverse down the first column and then down the second column etc
def _vectortoDFIndex(numRows, i):
    f = math.floor(i / numRows)
    r = i % numRows
    return [r, f]


# arguments: s = Sheet. y = dependent column name. x = independent column name.
# returns: results of univariate linear regression of y on x.
def reg(self, y, x, clean_data="greedy"):

    # prepare data
    v = np.array(x)
    v = np.append(v, y)
    dfClean = cleanData(self.df, v, clean_data)
    X = dfClean[x]
    y = dfClean[y]
    X = sm.add_constant(X)

    results = sm.OLS(y, X).fit()
    return results.summary()


# arguments: s = Sheet. y = dependent column name. x = independent column names.
# returns: results of multivariate linear regression of y on X.
def multiReg(self, y, X, clean_data="greedy"):
    
    # prepare data
    v = np.copy(X)
    v = np.append(v, y)
    dfClean = cleanData(self.df, v, clean_data)
    X = dfClean[X]
    y = dfClean[y]
    X = sm.add_constant(X)

    results = sm.OLS(y, X).fit()
    return results.summary()


# arguments: s = Sheet. y = dep var. x = ind var. id = entity identifier. year = time indentifier.
# returns: fit from fixed effects regression of y on x subject to parameters
# 
# notes: working only for a single x
def fixedEffects(self, y, x, id, year, entity_Effects=False, time_Effects=False, cov_Type='clustered', cluster_Entity=True, clean_data="greedy"):
    
    # prepare data
    v = np.copy(x)
    v = np.append(v, y)
    df = cleanData(self.df, v, clean_data)
    
    # set up panel and return fit
    df = df.set_index([id, year])

    mod = PanelOLS(df[y], df[x], entity_effects = entity_Effects, time_effects = time_Effects)
    return mod.fit(cov_type=cov_Type, cluster_entity=cluster_Entity)


# arguments: s = Sheet. y = binary variable. X = vector of column names.
# returns: logistic classification model fitting y to Xs
def logisticRegression(self, y, X, clean_data="greedy"):

    df = self.df

    # prepare data
    v = np.copy(X)
    v = np.append(v, y)
    dfClean = cleanData(df, v, clean_data)
    X = dfClean[X]
    y = dfClean[y]
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit()
    return model


# arguments: model from logisticRegression()
# returns: printed summary of model
def summarizeLogisticRegression(model):
    return model.summary()


# arguments: model from logisticRegression(). where = where in domain to apply. how = method of application.
# returns: marginal effects of model subject (kind of the derivative of logistic function, important for interpretation)
def logisticMarginalEffects(model, where="overall", how="dydx"):
    print(model.get_margeff(at=where,method=how).summary())


# arguments: s = Sheet. y = dep var name. X = group of ind vars. Z = group of instruments. exog_regressors = controls.
# returns: fit from instrumental variable regression
# 
# notes: GMM algorithm is allowed rather than 2SLS because it is more efficient for large-scale numerical
# optimization and computation. Still 2SLS is default because coefficient estimates can differe (I think 
# especially in small sample sizes) and 2SLS is the industry standard. GMM output format also seems to work
# better with instrument-exogeneity testing functions 
def ivRegress(self, y, X, Z, exog_regressors=-1, clean_data="greedy", covType="unadjusted", method="2SLS"):
    
    df = self.df

   # Check for sufficient first-stage identification 
    if type(X) is str:
        num_endogenous_regressors = 1
    else:
        num_endogenous_regressors = len(X)
    if type(Z) is str:
        num_instruments = 1
    else:
        num_instruments = len(Z)

    if (num_instruments < num_endogenous_regressors):
        print("Error: We need as many instruments as endogenous covariates for two-stage least squares.")
        return

    # prepare data
    v = np.copy(X)
    v = np.append(v, y)
    v = np.append(v, Z)
    if exog_regressors != -1:
        v = np.append(v, exog_regressors)
    dfClean = cleanData(df, v, clean_data)
    X = dfClean[X]
    length = len(X)
    Z = dfClean[Z]
    y = dfClean[y]
    if exog_regressors != -1:
        exog_regressors = sm.add_constant(dfClean[exog_regressors])
    else:
        exog_regressors = np.full((length,1), 1)

    if method == "2SLS":
        mod = IV2SLS(y, exog_regressors, X, Z)
    if method == "GMM":
        mod = IVGMM(y, exog_regressors, X, Z)

    return mod.fit(cov_type=covType)


# arguments: s = Sheet. y = dep var name. X = group of ind vars. Z = group of instruments. exog_regressors = controls.
# returns: exogeneity test for iv regression.
#  
# notes: Unfortunately until I think of something smarter we have to run the regression
# again and use generalized method of moment estimation for the coefficient computation
# Requires instrument overidentification
def homoskedasticJStatistic(self, y, X, Z, exog_regressors=-1, clean_data="greedy", covType="unadjusted"):

    # Check overidentification 
    if type(X) is str:
        num_endogenous_regressors = 1
    else:
        num_endogenous_regressors = len(X)
    if type(Z) is str:
        num_instruments = 1
    else:
        num_instruments = len(Z)

    if (num_instruments <= num_endogenous_regressors):
        print("Underidentification Error: We need more instruments than endogenous regressors for this test.")
        return

    df = self.df

    # prepare data
    v = np.copy(X)
    v = np.append(v, y)
    v = np.append(v, Z)
    if exog_regressors != -1:
        v = np.append(v, exog_regressors)
    dfClean = cleanData(df, v, clean_data)
    X = dfClean[X]
    length = len(X)
    Z = dfClean[Z]
    y = dfClean[y]
    if exog_regressors != -1:
        exog_regressors = sm.add_constant(dfClean[exog_regressors])
    else:
        exog_regressors = np.full((length,1), 1)

    mod = IVGMM(y, exog_regressors, X, Z)
    res = mod.fit()
    return res.j_stat


# arguments: s = Sheet. X = group of ind vars. Z = group of instruments.
# returns: test for joint strength of instruments
#
# notes: Motivation: instruments are asymptotically consistent but poorly behaved in normal-sized 
# samples with they are not very explanatory in the first stage. (The 2SLS coefficient boils
# down to a ratio of covariances.)
# At the moment this works only for a single endogenous covariate (but any number of instruments > 1)
# Will need to implement something called Anderson-Rubin algorithm later but it's going to be an
# absolute bitch
# If first-stage F-statistic < 10, this could indicate the presence of a weak instrument. Rotating
# out instruments can remove the weakness and get more consistent coefficient estimates in the
# second stage
def test_weak_instruments(self, x, Z, clean_the_data="greedy", covType="unadjusted"):

    # use multiReg because we just need first stage results
    results = self.multiReg(x, Z, clean_data=clean_the_data)

    # want F > 10
    return results.fvalue


# arguments: s = Sheet. y = var. dates = the dates for times series. p = number of lags in model
# returns: univariate (AR(p)) time series autoregression model.
def auto_reg(self, y, dates, p, clean_data="greedy"):

    v = np.copy(y)
    v = np.append(v, dates)

    # prepare data
    dfClean = cleanData(self.df, v, clean_data)
    time_series = dfClean[v]
    
    time_series = time_series.set_index(dates)
    model = s_ar.AR(time_series)
    results = model.fit(p)

    return results


# arguments: results from auto_reg() (AR() results wrapper)
# returns: prints results "nicely"
# 
# notes: this is kind of a mess because this module has been deprecated, so this function
# just prints a bunch of shit that's hopefully helpful. argument = AR results wrapper
def print_a_bunch_of_AR_shit(results):
    print()
    print("Model Parameters")
    print()
    print(results.params)
    print()
    print()
    print()
    print("Parameter Confidence Intervals")
    print()
    print(results.conf_int())
    print()
    print()
    print()
    print("Normalized Covariance Matrix Across Parameters")
    print()
    print(results.normalized_cov_params)
    print()
    print()


# arguments: s = Sheet. var = name of column. p = number of lags. ma = moving average parameter.
# returns: summary of ARMA regression, which can handle data that are slightly less stationary
def AR_with_moving_average(self, var, p, ma, the_dates, clean_data="greedy"):

    # prepare data
    dfClean = cleanData(self.df, var, clean_data)
    time_series = dfClean[var]

    arma = ARMA(time_series, np.array(p, ma), dates=the_dates)
    fit = arma.fit()
    return fit.summary()


# arguments: time_series = data variable.
# returns: "probability" that the time series is stationary
# 
# notes: null hypothesis is that the process is *not* stationary. uses the complex unit root test.
# returns p-value. So we can say the process is stationary. if and only if p < some alpha
def augmented_dicky_fuller_test(time_series, max_lag=-1):

    if max_lag == -1:
        vector = s_st.adfuller(time_series)
    else:
        vector = s_st.adfuller(time_series, maxlag=max_lag)

    return vector[1]


# arguments: s = Sheet. y = group of vars. dates = the dates for times series. p = number of lags in model
# returns: multivariate (VAR(p)) time series autoregression model.
def vector_auto_reg(self, y, dates, p, clean_data="greedy"):

    v = np.copy(y)
    v = np.append(v, dates)

    # prepare data
    dfClean = cleanData(self.df, v, clean_data)
    time_series = dfClean[y]
    dates = dfClean[dates]

    time_series = time_series.set_index(dates)

    # run pth-order VAR
    model = VAR(time_series)
    results = model.fit(p)

    return results


# arguments: results from vector_auto_reg()
# returns: prints results summary
def summarize_VAR(results):
    return results.summary()


# arguments: results from vector_auto_reg(). dep = dep variable. ind = vars alleged to "cause" dep
# returns: Granger p-value. 
# 
# notes: requires that dep and ind be subsets of results's column variables. 
# p-value uses F-statistic (chi-squared) test, as is convention.
def granger_p_value(results, dep, ind):
    
    r = results.test_causality(dep, ind, kind='f')
    return r.p_value


# arguments: results from vector_auto_reg(). dep = dep variable. ind = vars alleged to "cause" dep
# returns: fully Granger causality summary
# 
# notes: requires that dep and ind be subsets of results's column variables. 
# p-value uses F-statistic (chi-squared) test, as is convention.
def granger_causality_test(results, dep, ind):
    r = results.test_causality(dep, ind, kind='f')
    return r.summary()


# def autocorrelation():

# def autocovariance():

# arguments: s = Sheeet. v = vector of column names
# returns PCA object
def principle_component_analysis(self, v, clean_data="greedy"):

    # prepare data
    dfClean = cleanData(self.df, v, clean_data)
    data = dfClean[v]

    pca = PCA(data)

    return pca

# argument: pca object from method above
# returns: just prints a lot of the relevant fields
def print_PCA_wrapper(pca):

    print("factors")
    print(pca.factors)
    print()

    print("coefficients")
    print(pca.coeff)
    print()

    print("eigenvalues")
    print(pca.eigenvals)
    print()

    print("eigenvectors (ordered)")
    print(pca.eigenvecs)
    print()

    print("transformed data")
    print(pca.transformed_data)
    print()


# arguments: s = Sheet. endog = dep variable column name. exog = ind var col name.
# returns: summary of Poisson regression
def poisson_regression(self, endog, exog, clean_data="greedy"):

    # prepare data
    v = np.copy(exog)
    v = np.append(v, endog)
    dfClean = cleanData(self.df, v, clean_data)
    exog = sm.add_constant(dfClean[exog])
    endog = dfClean[endog]

    poisson = Poisson(endog, exog)
    fit = poisson.fit()
    return fit.summary()


# arguments: s = Sheet. endog = dep variable. k = number of regimes.
# returns: summary dynamic regression model
def markov_switching_regime_regression(self, endog, k, exog_vars=-1, clean_data="greedy"):

    # prepare data
    v = np.copy(endog)
    if exog_vars != -1:
        v = np.append(v, exog_vars)
        dfClean = cleanData(s.df, v, clean_data)
        endog = dfClean[endog]

    else:
        endog = self.df[endog]

    if exog_vars == -1:
        exog_vars = None
    else:
        exog_vars = dfClean[exog_vars]

    mr = MarkovRegression(endog, k, exog=exog_vars)
    fit = mr.fit()
    return fit.summary()

# find kth largest element in an unsorted array a not including 1s
def _kthLargest(a, k):
    aSorted = np.sort(a)
    n = len(a)
    for i in range(0, n):
        if (aSorted[i] == 1):
            index = i
            break

    return aSorted[index - k + 1]

# candidates must be NumPy array!
'''def find_plausible_instruments(s, y, X, snitch, lengthX, short_list=-1, numToReturn=10, k=0, exog_regressors=-1, clean_data="greedy"):
    
    if not np.isscalar(short_list):
        candidates = short_list
    
    else:
        candidates = "WRONG"
        cols = s.df.columns
        for i in range(len(col)):
            name = cols[i]
            if (name != y and name != snitch):
                for j in range(lengthX):
                    if (name == X)

            candidates = np.append(candidates, cols[i])




    num_candidates = len(candidates)
    if (numToReturn > num_candidates):
        numToReturn = num_candidates

    if type(X) is str:
        k = 1
    
    else:
        k = len(X)

    # eliminate those that are weak
    

    
    # find the j-stat p-value for each of them when left out
    j_p_array = np.zeros(num_candidates)

    for i in range(num_candidates):
        one_out = np.delete(candidates, i)
        one_out_with_snitch = np.append(one_out, snitch)
        j_p_array[i] = homoskedasticJStatistic(s, y, X, one_out_with_snitch).pval

    sorted_index = np.argsort(j_p_array)

    # return the best ones
    best_k_candidates = candidates[sorted_index[0]]
    for i in range(1, k):
        best_k_candidates = np.append(best_k_candidates, candidates[sorted_index[i]])

    return best_k_candidates

'''