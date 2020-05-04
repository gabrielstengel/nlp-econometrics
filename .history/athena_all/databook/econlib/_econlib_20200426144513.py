import math
from itertools import combinations

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

from athena_all.databook.queryresult import QueryResult

################### here to get rid of Warning messages ########################
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

################################################################################


######## THE FUNCTIONS THAT NEED SOME TLC ARE: AR_with_moving_average


class EconLibMixin:

    # ==================================================================================#
    # SUMMARY STATISTIC METHODS ========================================================#
    # ==================================================================================#

    # argument: string column name col
    # returns mean of column
    def findMean(self, col):
        mean = np.mean(self.get_column(col))
        utterance = "The mean of " + str(col) + " is " + str(mean) + "."
        return QueryResult(utterance, utterance)

    # argument: vector x
    # returns standard devation of x
    def findStd(self, col):
        std = np.std(self.get_column(col))
        utterance = "The standard deviation of " + str(col) + " is " + str(std) + "."
        return QueryResult(utterance, utterance)

    # argument: vector x
    # returns variance of x
    def findVar(self, col):
        var = np.var(self.get_column(col))
        utterance = "The variance of " + str(col) + " is " + str(var) + "."
        return QueryResult(utterance, utterance)

    # argument: vector x and vector y
    # returns covariance of x and y
    def findCoVar(self, x, y=None):

        if type(x) is list:
            var = np.cov([self.get_column(col) for col in x])
            utterance = (
                "The covariance matrix of the columns "
                + str(x)
                + "is "
                + str(var)
                + "."
            )

        var = np.cov(self.get_column(x), self.get_column(y))
        utterance = (
            "The covariance of "
            + str(x)
            + "and "
            + str(y)
            + "is "
            + str(var[0][1])
            + "."
        )
        return QueryResult(utterance, utterance)

    # argument: vector x
    # returns maximum value of x
    def findMax(self, col):
        s = self.map_column_to_sheet(col)
        df = s.df
        maximum = np.max(df[[col]])[0]
        utterance = "The maximum of " + str(col) + " is " + str(maximum) + "."
        return QueryResult(utterance, utterance)

    # argument: vector x
    # returns minimum value of x
    def findMin(self, col):
        s = self.map_column_to_sheet(col)
        df = s.df
        minimum = np.min(df[[col]])[0]
        utterance = "The minimum of " + str(col) + " is " + str(minimum) + "."
        return QueryResult(utterance, utterance)

    # argument: vector x
    # returns median value of x
    def findMedian(self, col):
        s = self.map_column_to_sheet(col)
        df = s.df
        median = np.median(df[[col]])
        utterance = "The median of " + str(col) + " is " + str(median) + "."
        return QueryResult(utterance, utterance)

    ############### SHOULD REALLY HAVE A COVARIANCE METHOD HERE ########################

    # ==================================================================================#
    # CORRELATION METHODS ==============================================================#
    # ==================================================================================#

    # arguments: s = Sheet. yColName and xColName = column names.
    # returns: their correlation
    def findCorr(self, yColName, xColName):
        s = self.map_column_to_sheet(yColName)
        corr = s.corr_matrix[yColName][s.findColumnIndexGivenName(xColName)]
        utterance = (
            "The correlation between "
            + str(yColName)
            + " and "
            + str(xColName)
            + " is "
            + str(corr)
            + "."
        )
        return QueryResult(corr, utterance)

    # arguments: s = Sheet. col = name of column.
    # returns: name of column in s's df most correlated with col.
    def largestCorr(self, col):

        s = self.map_column_to_sheet(col)
        m = s.corr_matrix
        col_corrs = m[[col]]
        champ = -1
        max = -2

        for i in range(0, len(col_corrs)):
            c = abs(col_corrs.iloc[i, 0])
            if c > max and c < 1:
                max = c
                champ = i

        if champ == -1:
            utterance = "ERROR: none found"
            return QueryResult(None, utterance)

        best_var = s.findVariableName(m, champ)
        utterance = (
            "The variable most correlated with "
            + str(col)
            + " is "
            + str(best_var)
            + "."
        )
        return QueryResult(best_var, utterance)

    # arguments: s = Sheet. col = name of column. num_return = number seeked to return.
    # returns: names of num_return columns in s's df most correlated with col.
    def largestCorrList(self, col, num_return=3):

        s = self.map_column_to_sheet(col)
        df = s.df
        m = s.corr_matrix
        v = abs(m[[col]])
        n = len(v)
        vAbs = np.zeros(n)

        for i in range(0, n):
            vAbs[i] = v.iloc[i, 0]

        # we want the (num_return + 1)'th largest because we don't it to count itself
        p = _kthLargest(vAbs, num_return + 1)

        returnVector = []

        for i in range(0, n):
            c = vAbs[i]
            if c >= p and c < 1:
                returnVector.append(s.findVariableName(m, i))

        utterance = (
            "The "
            + str(num_return)
            + " variables most correlated with "
            + str(col)
            + " are "
        )
        for i in range(0, len(returnVector) - 1):
            utterance = utterance + returnVector[i] + ", "

        utterance = utterance + "and " + returnVector[len(returnVector) - 1] + "."

        return QueryResult(returnVector, utterance)

    # arguments: s = Sheet. num_return = number seeked for return.
    # returns: largest num_return pairwise correlations overall in the dataset (string column names)
    def overallLargestCorrs(self, num_return=5, index=0):

        s = self.sheets[index]
        m = s.corr_matrix
        n = len(m)

        v = np.zeros(n * n)

        for i in range(0, n):
            for j in range(i, n):
                element = abs(m.iloc[i, j])
                v[_dfToVectorIndex(n, i, j)] = element

        p = _kthLargest(v, num_return + 1)
        r = np.zeros(num_return)
        j = 0

        for i in range(0, len(v)):
            vi = v[i]
            if vi >= p and vi < 1:
                r[j] = i
                j += 1

        returnVector = []

        for i in range(0, len(r)):
            ri = r[i]
            t = _vectortoDFIndex(n, ri)
            returnVector.append(s.findVariableName(m, t))

        utterance = (
            "Here are the "
            + str(num_return)
            + " most correlative pairwise relationships in the dataset\n"
        )
        utterance = utterance + str(returnVector)

        return QueryResult(returnVector, utterance)

    # ==================================================================================#
    # SIMPLE REGRESSION METHODS ========================================================#
    # ==================================================================================#

    # arguments: s = Sheet. y = dependent column name. x = independent column name.
    # returns: results of univariate linear regression of y on x.
    def reg(self, y, x, clean_data="greedy"):

        s = self.map_column_to_sheet(y)
        y_arg = y
        x_arg = x

        # prepare data
        v = np.array(x)
        v = np.append(v, y)
        dfClean = s.cleanData(v, clean_data)
        X = dfClean[x]
        y = dfClean[y]
        X = sm.add_constant(X)

        results = sm.OLS(y, X).fit()
        utterance = (
            "Here are the results of a linear regression of "
            + str(y_arg)
            + " on "
            + str(x_arg)
            + ".\n\n"
        )
        utterance = utterance + str(results.summary())

        return QueryResult(results.summary(), utterance)

    # arguments: s = Sheet. y = dependent column name. x = independent column names.
    # returns: results of multivariate linear regression of y on X.
    def multiReg(self, y, X, clean_data="greedy"):

        s = self.map_column_to_sheet(y)
        y_arg = y
        X_arg = X

        # prepare data
        v = np.copy(X)
        v = np.append(v, y)
        dfClean = s.cleanData(v, clean_data)
        X = dfClean[X]
        y = dfClean[y]
        X = sm.add_constant(X)

        results = sm.OLS(y, X).fit()

        utterance = (
            "Here are the results of a multivariate linear regression of "
            + str(y_arg)
            + " on "
            + str(X_arg)
            + ".\n\n"
        )
        utterance = utterance + str(results.summary())

        return QueryResult(results.summary(), utterance)

    # arguments: s = Sheet. y = dep var. x = ind var. id = entity identifier. year = time indentifier.
    # returns: fit from fixed effects regression of y on x subject to parameters
    #
    # notes: working only for a single x
    def fixedEffects(
        self,
        y,
        x,
        id,
        year,
        entity_Effects=False,
        time_Effects=False,
        cov_Type="clustered",
        cluster_Entity=True,
        clean_data="greedy",
    ):

        if type(x) != str:
            utterance = (
                "ERROR: Multiple independent regressor approach not yet implemented."
            )
            return utterance

        s = self.map_column_to_sheet(y)

        # prepare data
        v = np.copy(x)
        v = np.append(v, y)
        df = s.cleanData(v, clean_data)

        # set up panel and return fit
        df = df.set_index([id, year])

        mod = PanelOLS(
            df[y], df[x], entity_effects=entity_Effects, time_effects=time_Effects
        )
        utterance = (
            "Here are the results of a fixed effects regression of "
            + str(y)
            + " on "
            + str(x)
        )
        utterance = (
            utterance
            + ", using "
            + str(year)
            + " as the time dimension and "
            + str(id)
            + " as the id dimension.\n\n"
        )
        utterance = utterance + str(
            mod.fit(cov_type=cov_Type, cluster_entity=cluster_Entity)
        )

        return QueryResult(
            mod.fit(cov_type=cov_Type, cluster_entity=cluster_Entity), utterance
        )

    # arguments: model from logisticRegression()
    # returns: printed summary of model
    def summarizeLogisticRegression(self, y, X, clean_data="greedy"):

        s = self.map_column_to_sheet(y)
        model = logisticRegression(y, X, s, clean_data=clean_data)

        utterance = (
            "Here is a summary of the results of a logistic regression of " + str(y)
        )
        utterance = utterance + " on " + str(X) + "\n\n" + str(model.summary())

        return QueryResult(model.summary(), utterance)

    # arguments: model from logisticRegression(). where = where in domain to apply. how = method of application.
    # returns: marginal effects of model subject (kind of the derivative of logistic function, important for interpretation)
    def logisticMarginalEffects(
        self, y, X, where="overall", how="dydx", clean_data="greedy"
    ):

        s = self.map_column_to_sheet(y)
        model = logisticRegression(y, X, s, clean_data=clean_data)

        utterance = (
            "Here are the marginal effects evaluated at "
            + str(where)
            + " using the method "
        )
        utterance = (
            utterance
            + str(how)
            + " of a logistic regression of "
            + str(y)
            + " on "
            + str(X)
        )
        utterance = (
            utterance + "\n\n" + str(model.get_margeff(at=where, method=how).summary())
        )

        return QueryResult(model.get_margeff(at=where, method=how).summary(), utterance)

    # ==================================================================================#
    # INSTRUMENTAL VARIABLE METHODS ====================================================#
    # ==================================================================================#

    # arguments: s = Sheet. y = dep var name. X = group of ind vars. Z = group of instruments. exog_regressors = controls.
    # returns: fit from instrumental variable regression
    #
    # notes: GMM algorithm is allowed rather than 2SLS because it is more efficient for large-scale numerical
    # optimization and computation. Still 2SLS is default because coefficient estimates can differe (I think
    # especially in small sample sizes) and 2SLS is the industry standard. GMM output format also seems to work
    # better with instrument-exogeneity testing functions
    def ivRegress(
        self,
        y,
        X,
        Z,
        exog_regressors=-1,
        clean_data="greedy",
        covType="unadjusted",
        method="2SLS",
    ):
        try:
            s = self.map_column_to_sheet(y)
            df = s.df
            y_args = y
            X_args = X
            Z_args = Z

            # Check for sufficient first-stage identification
            if type(X) is str:
                num_endogenous_regressors = 1
            else:
                num_endogenous_regressors = len(X)
            if type(Z) is str:
                num_instruments = 1
            else:
                num_instruments = len(Z)

            if num_instruments < num_endogenous_regressors:
                utterance = "Error: We need at least as many instruments as endogenous covariates for two-stage least squares."
                return QueryResult(None, utterance)

            # prepare data
            v = np.copy(X)
            v = np.append(v, y)
            v = np.append(v, Z)
            if exog_regressors != -1:
                v = np.append(v, exog_regressors)
            dfClean = s.cleanData(v, clean_data)
            X = dfClean[X]
            length = len(X)
            Z = dfClean[Z]
            y = dfClean[y]
            if exog_regressors != -1:
                exog_regressors = sm.add_constant(dfClean[exog_regressors])
            else:
                exog_regressors = np.full((length, 1), 1)

            if method == "2SLS":
                mod = IV2SLS(y, exog_regressors, X, Z)
            if method == "GMM":
                mod = IVGMM(y, exog_regressors, X, Z)

            utterance = "Here are the results of an regression of " + str(y_args)
            utterance = (
                utterance
                + " on endogenous covariates "
                + str(X_args)
                + ", using "
                + str(Z_args)
                + " as instruments.\n\n"
            )
            utterance = utterance + str(mod.fit(cov_type=covType))

            return QueryResult(mod.fit(cov_type=covType), utterance)

        except:
            return QueryResult(None, None)

    # arguments: s = Sheet. y = dep var name. X = group of ind vars. Z = group of instruments. exog_regressors = controls.
    # returns: exogeneity test for iv regression.
    #
    # Requires instrument overidentification
    def homoskedasticJStatistic(
        self, y, X, Z, exog_regressors=-1, clean_data="greedy", covType="unadjusted"
    ):

        s = self.map_column_to_sheet(y)
        df = s.df

        arg_y = y
        arg_X = X
        arg_Z = Z

        # Check overidentification
        if type(X) is str:
            num_endogenous_regressors = 1
        else:
            num_endogenous_regressors = len(X)
        if type(Z) is str:
            num_instruments = 1
        else:
            num_instruments = len(Z)

        if num_instruments <= num_endogenous_regressors:
            utterance = "Underidentification Error: We need more instruments than endogenous regressors for this test."
            return QueryResult(None, utterance)

        # prepare data
        v = np.copy(X)
        v = np.append(v, y)
        v = np.append(v, Z)
        if exog_regressors != -1:
            v = np.append(v, exog_regressors)
        dfClean = s.cleanData(v, clean_data)
        X = dfClean[X]
        length = len(X)
        Z = dfClean[Z]
        y = dfClean[y]
        if exog_regressors != -1:
            exog_regressors = sm.add_constant(dfClean[exog_regressors])
        else:
            exog_regressors = np.full((length, 1), 1)
        mod = IVGMM(y, exog_regressors, X, Z)
        res = mod.fit()
        j_stat = res.j_stat

        utterance = (
            "\nThe homoskedastic Wald j-statistic test output in a regression of "
            + str(arg_y)
            + " on endogenous covariates "
            + str(arg_X)
        )
        utterance = (
            utterance
            + ", using "
            + str(arg_Z)
            + " as instruments is the following:\n\n"
        )
        utterance = utterance + str(j_stat) + "\n\n"

        return QueryResult(j_stat, utterance)

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
    def test_weak_instruments(
        self, x, Z, clean_the_data="greedy", covType="unadjusted"
    ):

        if type(x) != str:
            utterance = (
                "Multiple endogenous regressors not yet implemented for this test."
            )
            return QueryResult(None, utterance)

        s = self.map_column_to_sheet(x)

        x_arg = x
        Z_arg = Z

        # prepare data. use OLS because we just need first stage results
        v = np.copy(x)
        v = np.append(v, Z)
        dfClean = s.cleanData(v, clean_the_data)
        x = dfClean[x]
        Z = dfClean[Z]

        results = sm.OLS(x, Z).fit()

        # want F > 10
        f = results.fvalue
        utterance = (
            "The F-value in a regression with endogenous covariate "
            + str(x_arg)
            + " and instruments "
            + str(Z_arg)
        )
        utterance = (
            utterance
            + " is "
            + str(f)
            + ".\n Typically, we want F > 10 to have reliably strong estimates in the first stage."
        )

        return QueryResult(f, utterance)

    # arguments: y = dependent. X = group of independent vars.
    # exog_Z = instrument known to be valid. candidates = array of candidates for valid instruments
    # returns: optimal group of valid candidates
    #
    # DOUBLE CHECK WE'RE ON THE RIGHT SIDE OF THE P-VAL
    def find_instruments(self, y, X, exog_Z, candidates):

        s = self.map_column_to_sheet(y)
        df = s.df

        if np.isscalar(X):
            k = 1
        else:
            k = len(X)

        if np.isscalar(candidates):
            print("Multiple instrument candidates required.")
        else:
            n = len(candidates)

        utterance = (
            "This is the most inclusive group of jointly valid instruments among those"
        )
        utterance = (
            utterance
            + " suggested in a regression of "
            + str(y)
            + " on "
            + str(X)
            + " with "
            + str(exog_Z)
        )
        utterance = utterance + " as a known exogenous instrument:\n\n"

        # base case: test them all
        Z_with_exog = np.append(candidates, exog_Z)
        j_pval = self.homoskedasticJStatistic(y, X, Z_with_exog).get_denotation().pval
        if j_pval > 0.05:
            return QueryResult(candidates, utterance + str(candidates))

        turn = 1
        a = np.arange(n)
        while turn <= n - k:

            combs = combinations(a, n - turn)

            Z = np.empty(n - turn, dtype=object)

            for group in list(combs):
                for i in range(len(group)):
                    Z[i] = candidates[group[i]]

                Z_with_exog = np.append(Z, exog_Z)
                j_pval = (
                    self.homoskedasticJStatistic(y, X, Z_with_exog)
                    .get_denotation()
                    .pval
                )

                if j_pval > 0.05:
                    return QueryResult(Z, utterance + str(Z))

            turn += 1

        return QueryResult(None, "No valid groups of instruments found.")

    # ==================================================================================#
    # TIME SERIES METHODS ==============================================================#
    # ==================================================================================#

    # arguments: results from auto_reg() (AR() results wrapper)
    # returns: prints results "nicely"
    #
    # notes: this is kind of a mess because this module has been deprecated, so this function
    # just prints a bunch of shit that's hopefully helpful. argument = AR results wrapper
    def print_a_bunch_of_AR_shit(self, y, dates, p, clean_data="greedy"):

        s = self.map_column_to_sheet(y)
        results = auto_reg(y, dates, p, s, clean_data=clean_data)

        utterance = (
            "Here are the results of a univariate time series on " + str(y) + " with "
        )
        utterance = utterance + str(p) + " lags:\n\n\n"
        utterance = utterance + "Model Parameters:\n" + str(results.params) + "\n\n\n"
        utterance = (
            utterance
            + "Parameter Confidence Intervals:\n"
            + str(results.conf_int())
            + "\n\n\n"
        )
        utterance = (
            utterance
            + "Normalized Covariance Matrix Across Parameters:\n"
            + str(results.normalized_cov_params)
            + "\n\n\n"
        )

        # train just on parameters
        return QueryResult(str(results.params), utterance)

    # arguments: s = Sheet. var = name of column. p = number of lags. ma = moving average parameter.
    # returns: summary of ARMA regression, which can handle data that are slightly less stationary
    def AR_with_moving_average(self, var, p, ma, the_dates, clean_data="greedy"):

        s = self.map_column_to_sheet(var)

        # prepare data
        dfClean = s.cleanData(var, clean_data)
        time_series = dfClean[var]

        arma = ARMA(time_series, np.array(p, ma), dates=the_dates)
        fit = arma.fit()

        utterance = (
            "Here are the results of an ARMA regression of "
            + str(var)
            + " in "
            + str(the_dates)
            + ".\n"
        )
        return QueryResult(fit.summary(), utterance + str(fit.summary()))

    # arguments: time_series = data variable.
    # returns: "probability" that the time series is stationary
    #
    # notes: null hypothesis is that the process is *not* stationary. uses the complex unit root test.
    # returns p-value. So we can say the process is stationary. if and only if p < some alpha
    def augmented_dicky_fuller_test(self, var, max_lag=-1):

        s = self.map_column_to_sheet(var)
        df = s.df
        time_series = df[var]

        if max_lag == -1:
            vector = s_st.adfuller(time_series)
        else:
            vector = s_st.adfuller(time_series, maxlag=max_lag)

        utterance = "Here is the p-value of an augmented Dicky-Fuller (stationarity) test with variable "
        utterance = utterance + str(var) + "."
        utterance = (
            utterance
            + "The null hypothesis is that the process has a unit root. The lower the p-value, "
        )
        utterance = utterance + "the stronger the case for stationarity.\n"
        utterance = utterance + str(vector[1])

        return QueryResult(vector[1], utterance)

    # arguments: results from vector_auto_reg()
    # returns: prints results summary
    def summarize_VAR(self, y, dates, p, clean_data="greedy"):

        s = self.map_column_to_sheet(y[0])

        results = vector_auto_reg(y, dates, p, s, clean_data=clean_data)

        utterance = (
            "Here are the results of a vector autoregression with the variables "
            + str(y)
        )
        utterance = utterance + " with " + str(p) + " lags:\n\n"
        utterance = utterance + str(results.summary())
        return QueryResult(results.summary(), utterance)

    # arguments: results from vector_auto_reg(). dep = dep variable. ind = vars alleged to "cause" dep
    # returns: fully Granger causality summary
    #
    # notes: requires that dep and ind be subsets of results's column variables.
    # p-value uses F-statistic (chi-squared) test, as is convention.
    def granger_causality_test(self, y, dates, p, dep, clean_data="greedy"):

        s = self.map_column_to_sheet(y[0])
        results = vector_auto_reg(y, dates, p, s, clean_data=clean_data)

        ind = np.empty(len(y) - 1, dtype=object)
        j = 0
        for i in range(len(y)):
            if y[i] != dep:
                ind[j] = y[i]
                j += 1

        r = results.test_causality(dep, ind)
        utterance = (
            "Here is a summary of the results of the Granger causality test of "
            + str(ind)
        )
        utterance = utterance + " on " + str(dep) + " with " + str(p) + " lags:\n\n"
        return QueryResult(r.summary(), utterance + str(r.summary()))

    # arguments: cols = vector of one or more column names. time = column of time index.
    # min_lag and max_lag = domain of consideration
    # returns: a string that gives in-depth consideration of model choice in a vector autoregression model
    def analyze_lags(self, cols, time, preferred_criterion="aic", min_lag=1, max_lag=3):

        try:
            s = self.map_column_to_sheet(cols)
            multi = False
        except:
            s = self.map_column_to_sheet(cols[0])
            multi = True

        df = s.df

        if multi:
            try:
                args_vector = np.append(cols, time)
                data = df[args_vector]
                data = data.set_index(time)
            except:
                data = df[cols]

            model = VAR(data)

        else:
            try:
                args_vector = np.array([cols, time])
                data = df[args_vector]
                data = data.set_index(time)
            except:
                data = df[cols]

            model = s_ar.AR(data)

        aic = np.zeros(max_lag - min_lag + 1)
        bic = np.zeros(max_lag - min_lag + 1)

        for i in range(max_lag - min_lag + 1):
            fit = model.fit(i + min_lag)
            try:
                aic[i] = fit.aic
                bic[i] = fit.bic
            except:
                utterance = "Error in Cholesky decomposition. (Positive definite specification not met.)"
                utterance = (
                    utterance + " Try decreasing the maximum number of lags considered."
                )
                return QueryResult(None, utterance)

        utterance = ""
        for i in range(max_lag - min_lag + 1):
            utterance = (
                utterance + "AIC (" + str(i + min_lag) + " lags): " + str(aic[i]) + "\n"
            )

        utterance = utterance + "\n\n"

        for i in range(max_lag - min_lag + 1):
            utterance = (
                utterance + "BIC (" + str(i + min_lag) + " lags): " + str(bic[i]) + "\n"
            )

        utterance = utterance + "\n\n"

        x = np.argsort(aic)
        champ = aic[x[0]]
        utterance = (
            utterance
            + "Using AIC, here are the estimated proportional probabilities, using the best as a reference:"
        )
        utterance = utterance + "\n"
        for i in range(max_lag - min_lag + 1):
            utterance = (
                utterance
                + str(i + min_lag)
                + " lags: "
                + str(find_prob_given_AIC(champ, aic[i]))
                + "\n"
            )

        optimal = find_optimal_lag_length(
            cols,
            time,
            s,
            multi,
            min_lag=min_lag,
            max_lag=max_lag,
            criterion=preferred_criterion,
        ).get_denotation()

        return QueryResult(optimal, utterance)

    # def autocorrelation():

    # def autocovariance():

    # ==================================================================================#
    # ADVANCED REGRESSION METHODS ======================================================#
    # ==================================================================================#

    # argument: pca object from method above
    # returns: just prints a lot of the relevant fields
    def print_PCA_wrapper(self, v, clean_data="greedy"):

        s = self.map_column_to_sheet(v[0])

        pca = principle_component_analysis(v, s, clean_data=clean_data)

        utterance = (
            "Here are the results of a Principle Component Analysis with the variables "
            + str(v)
        )
        utterance = utterance + ":\n\n"

        utterance = utterance + "factors:\n"
        utterance = utterance + str(pca.factors) + "\n"

        utterance = utterance + "coefficients:\n"
        utterance = utterance + str(pca.coeff) + "\n"

        utterance = utterance + "eigenvalues:\n"
        utterance = utterance + str(pca.eigenvals) + "\n"

        utterance = utterance + "eigenvectors (ordered):\n"
        utterance = utterance + str(pca.eigenvecs) + "\n"

        utterance = utterance + "transformed data:\n"
        utterance = utterance + str(pca.transformed_data) + "\n"

        return QueryResult(str(pca.coeff), utterance)

    # arguments: s = Sheet. endog = dep variable column name. exog = ind var col name.
    # returns: summary of Poisson regression
    def poisson_regression(self, endog, exog, clean_data="greedy"):

        s = self.map_column_to_sheet(endog)

        arg_endog = endog
        arg_exog = exog

        # prepare data
        v = np.copy(exog)
        v = np.append(v, endog)
        dfClean = s.cleanData(v, clean_data)
        exog = sm.add_constant(dfClean[exog])
        endog = dfClean[endog]

        poisson = Poisson(endog, exog)
        fit = poisson.fit()

        utterance = (
            "Here are the results of a Poisson regression with endogenous variables "
        )
        utterance = (
            utterance
            + str(arg_endog)
            + " and exogenous variables "
            + str(arg_exog)
            + ".\n"
        )
        utterance = utterance + str(fit.summary())

        return QueryResult(fit.summary(), utterance)

    # arguments: s = Sheet. endog = dep variable. k = number of regimes.
    # returns: summary dynamic regression model
    def markov_switching_regime_regression(
        self, endog, k, exog_vars=-1, clean_data="greedy"
    ):

        s = self.map_column_to_sheet(endog)

        arg_endog = endog

        # prepare data
        v = np.copy(endog)
        if exog_vars != -1:
            v = np.append(v, exog_vars)
            dfClean = s.cleanData(v, clean_data)
            endog = dfClean[endog]

        else:
            endog = s.df[endog]

        if exog_vars == -1:
            exog_vars = None
        else:
            exog_vars = dfClean[exog_vars]

        mr = MarkovRegression(endog, k, exog=exog_vars)
        fit = mr.fit()
        utterance = (
            "Here are the results of a dynamic Markov regression with endogenous variable "
            + str(arg_endog)
        )
        utterance = utterance + " and " + str(k) + " regimes.\n"
        utterance = utterance + str(fit.summary())

        return QueryResult(fit.summary(), utterance)

    # arguments: sets_of_vars is an array of arrays. Each element is a model. Requires that they are on the same sheet.
    # first one is the dependent??
    # returns: the best of the models.
    def choose_among_regression_models(
        self, sets_of_vars, criterion="aic", clean_data="greedy"
    ):

        try:
            n = len(sets_of_vars)
            if n == 1:
                print("Multiple sets of variables required.")
                return
        except:
            print("Multiple sets of variables required.")
            return

        info_loss = np.zeros(n)

        for i in range(n):
            v = sets_of_vars[i]

            try:
                s = self.map_column_to_sheet(v[0])
            except:
                print("ERROR: One or more variables not in dataset.")
                return

            if len(v) == 1:
                print("Multiple variables per set required.")
                return

            y = v[0]
            X = np.array([v[1]])

            for j in range(1, len(v) - 1):
                X = np.append(X, v[j + 1])

            # prepare data
            t = np.array(X)
            t = np.append(t, y)
            dfClean = s.cleanData(t, clean_data)
            X = dfClean[X]
            y = dfClean[y]

            X = sm.add_constant(X)

            fit = sm.OLS(y, X).fit()
            if criterion == "aic":
                info_loss[i] = fit.aic
            elif criterion == "bic":
                info_loss[i] = fit.bic
            else:
                return QueryResult(None, "ERROR: Criterion argument not supported.")

        x = np.argsort(info_loss)
        best = sets_of_vars[x[0]]
        utterance = (
            "The best-fitting model according to the "
            + str(criterion)
            + " criterion among these ones:\n"
        )
        utterance = utterance + str(sets_of_vars) + "\n\n"
        utterance = utterance + "is:\n" + str(best)

        return QueryResult(best, utterance)

    # ==================================================================================#
    # END OF ECONOMICS LIBRARY =========================================================#
    # ==================================================================================#

    def get_econlib_fmap(self):

        return [
            # ===================== Summary Statistic Methods ===========================#
            (["mean", "average", "avg"], "findMean", self.findMean),
            (["variance", "var", "covar", "covariance"], "findCoVar", self.findCoVar),
            (
                [
                    "std",
                    "standard deviation",
                    "standard dev",
                    "standarddev",
                    "deviation",
                    "stddev",
                ],
                "findStd",
                self.findStd,
            ),
            (["variance", "var", "spread"], "findVar", self.findVar),
            (["max", "maximum", "biggest", "largest"], "findMax", self.findMax),
            (["min", "minimum", "smallest"], "findMin", self.findMin),
            (["median", "middle"], "findMedian", self.findMedian),
            # ========================= Correlation Methods =============================#
            (
                [
                    "correlation",
                    "corr",
                    "relationship",
                    "pearson",
                    "related",
                    "relate",
                    "relates",
                    "correlates",
                    "correlate",
                    "correlated",
                ],
                "findCorr",
                self.findCorr,
            ),
            (
                [
                    "correlation",
                    "corr",
                    "relationship",
                    "relationships",
                    "pearson",
                    "related",
                    "relate",
                    "relates",
                    "correlates",
                    "correlate",
                    "correlated",
                ],
                "largestCorr",
                self.largestCorr,
            ),
            (
                [
                    "correlation",
                    "corr",
                    "relationship",
                    "relationships",
                    "pearson",
                    "related",
                    "relate",
                    "relates",
                    "correlates",
                    "correlate",
                    "correlated",
                    "correlations",
                ],
                "largestCorrList",
                self.largestCorrList,
            ),
            (
                [
                    "correlation",
                    "corr",
                    "relationship",
                    "relationships",
                    "pearson",
                    "related",
                    "relate",
                    "relates",
                    "correlates",
                    "correlate",
                    "correlated",
                    "correlations",
                    "structure",
                ],
                "overallLargestCorrs",
                self.overallLargestCorrs,
            ),
            # ===================== Simple Regression Methods ===========================#
            (
                [
                    "linear regression",
                    "reg",
                    "regression",
                    "regress",
                    "OLS",
                    "ordinary least squares",
                ],
                "reg",
                self.reg,
            ),
            (
                [
                    "multivariate regression",
                    "multivariable reg",
                    "reg",
                    "regress",
                    "linear regression",
                    "regression",
                    "regressed",
                    "OLS",
                    "ordinary least squares",
                ],
                "multiReg",
                self.multiReg,
            ),
            (
                [
                    "fixed effects",
                    "fixed effect",
                    "panel",
                    "longitudinal",
                    "entity",
                    "include dummies",
                    "including dummies",
                    "id",
                ],
                "fixedEffects",
                self.fixedEffects,
            ),
            (
                [
                    "logistic",
                    "logit",
                    "binary",
                    "logistic",
                    "classification",
                    "dummy",
                    "discrete",
                    "discrete choice",
                ],
                "summarizeLogisticRegression",
                self.summarizeLogisticRegression,
            ),
            (
                ["marginal effects", "margins",],
                "logisticMarginalEffects",
                self.logisticMarginalEffects,
            ),
            # ===================== Instrument Regression Methods ======================#
            (
                [
                    "instrument",
                    "instruments",
                    "iv",
                    "instrumental",
                    "instrumental variable",
                    "regress",
                    "regression",
                    "reg",
                    "2SLS",
                    "two stage least squares",
                    "two-stage least squares",
                ],
                "ivRegress",
                self.ivRegress,
            ),
            (
                [
                    "exogeneity",
                    "exogenous",
                    "j-statistic",
                    "valid",
                    "j stat",
                    "homoskedastic",
                    "homoscedastic",
                    "overidentification",
                    "j statistic",
                ],
                "homoskedasticJStatistic",
                self.homoskedasticJStatistic,
            ),
            (
                ["weak", "strong", "strength", "relevant", "relevance"],
                "test_weak_instruments",
                self.test_weak_instruments,
            ),
            (
                [
                    "best instruments",
                    "best instrumental variables",
                    "instruments are valid",
                    "valid instruments",
                    "instrumental variables are valid",
                    "valid instrumental variables",
                ],
                "find_instruments",
                self.find_instruments,
            ),
            # ======================== Time Series Methods =============================#
            (
                [
                    "time series regression",
                    "autoregression",
                    "ar",
                    "autoregress",
                    "autoreg",
                ],
                "print_a_bunch_of_AR_shit",
                self.print_a_bunch_of_AR_shit,
            ),
            (
                ["time series", "ARMA", "moving average", "floating average"],
                "AR_with_moving_average",
                self.AR_with_moving_average,
            ),
            (
                [
                    "augmented",
                    "dickey-fuller",
                    "unit root",
                    "stationary",
                    "stationarity",
                    "dickey",
                    "fuller",
                    "dickey",
                ],
                "augmented_dicky_fuller_test",
                self.augmented_dicky_fuller_test,
            ),
            (
                ["vector autoregression", "var", "time series regression",],
                "summarize_VAR",
                self.summarize_VAR,
            ),
            (
                [
                    "granger cause",
                    "granger causes",
                    "caused",
                    "causality",
                    "granger causality",
                    "causation",
                    "granger causation",
                    "granger caused",
                    "autoregression",
                    "time series regression",
                    "granger",
                    "granger caused",
                ],
                "granger_causality_test",
                self.granger_causality_test,
            ),
            (
                [
                    "should I use",
                    "time series",
                    "number of",
                    "how many",
                    "optimal",
                    "lags",
                    "criterion",
                    "how many",
                    "right number",
                    "lagged values",
                    "lag",
                    "analyze",
                ],
                "analyze_lags",
                self.analyze_lags,
            ),
            # ==================== Advanced Regression Methods =========================#
            (
                [
                    "principle",
                    "component",
                    "analysis",
                    "pca",
                    "eigenvalue decomposition",
                ],
                "print_PCA_wrapper",
                self.print_PCA_wrapper,
            ),
            (
                [
                    "poisson",
                    "poisson regression",
                    "poisson regress",
                    "distributed poisson",
                    "poisson reg",
                ],
                "poisson_regression",
                self.poisson_regression,
            ),
            (
                ["markov", "hidden", "hmm",],
                "markov_switching_regime_regression",
                self.markov_switching_regime_regression,
            ),
            (
                [
                    "model",
                    "models",
                    "best model",
                    "model is best",
                    "best fitting",
                    "most accurate",
                    "optimal",
                    "truest model",
                    "best fit",
                    "choose among",
                    "should i use",
                    "should i choose",
                    "out of sample",
                    "performs best",
                    "performs the best",
                    "perform best",
                    "perform the best",
                    "generalizable",
                    "generalizability",
                ],
                "choose_among_regression_models",
                self.choose_among_regression_models,
            ),
        ]
        # ==============================================================================#


# Just a list of the functions
"""
    # ==================================================================================#
    # SUMMARY STATISTIC METHODS ========================================================#
    # ==================================================================================#

    def findMean(self, col):

    def findStd(self, col):

    def findVar(self, col):

    def findMax(self, col):

    def findMin(self, col):

    def findMedian(self, col):

    # ==================================================================================#
    # CORRELATION METHODS ==============================================================#
    # ==================================================================================#

    def findCorr(self, yColName, xColName):

    def largestCorr(self, col):

    def largestCorrList(self, col, num_return=3):

    def overallLargestCorrs(self, num_return=5, index=0):

    # ==================================================================================#
    # SIMPLE REGRESSION METHODS ========================================================#
    # ==================================================================================#

    def reg(self, y, x, clean_data="greedy"):

    def multiReg(self, y, X, clean_data="greedy"):

    def fixedEffects():

    def summarizeLogisticRegression(self, y, X, clean_data="greedy"):

    def logisticMarginalEffects(self, model, where="overall", how="dydx"):

    # ==================================================================================#
    # INSTRUMENTAL VARIABLE METHODS ====================================================#
    # ==================================================================================#

    def ivRegress():

    def homoskedasticJStatistic(self, y, X, Z, exog_regressors=-1, clean_data="greedy", covType="unadjusted"):

    def test_weak_instruments(self, x, Z, clean_the_data="greedy", covType="unadjusted"):

    def find_instruments(self, y, X, exog_Z, candidates):

    # ==================================================================================#
    # TIME SERIES METHODS ==============================================================#
    # ==================================================================================#

    def print_a_bunch_of_AR_shit():

    def AR_with_moving_average(self, var, p, ma, the_dates, clean_data="greedy"):

    def augmented_dicky_fuller_test(self, var, max_lag=-1):
    
    def summarize_VAR():

    def granger_causality_test():

    def analyze_lags(self, cols, time, preferred_criterion="aic", min_lag=1, max_lag=8):

    # ==================================================================================#
    # ADVANCED REGRESSION METHODS ======================================================#
    # ==================================================================================#

    def print_PCA_wrapper():

    def poisson_regression(self, endog, exog, clean_data="greedy"):

    def markov_switching_regime_regression(self, endog, k, exog_vars=-1, clean_data="greedy"):

    def choose_among_regression_models(self, sets_of_vars, criterion="aic", clean_data="greedy":
        """


# arguments: s = Sheet. y = binary variable. X = vector of column names.
# returns: logistic classification model fitting y to Xs
def logisticRegression(y, X, s, clean_data="greedy"):

    # prepare data
    v = np.copy(X)
    v = np.append(v, y)
    dfClean = s.cleanData(v, clean_data)
    X = dfClean[X]
    y = dfClean[y]
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit()
    return model


# arguments: s = Sheet. y = group of vars. dates = the dates for times series. p = number of lags in model
# returns: multivariate (VAR(p)) time series autoregression model.
def vector_auto_reg(y, dates, p, s, clean_data="greedy"):

    v = np.copy(y)
    v = np.append(v, dates)

    # prepare data
    dfClean = s.cleanData(v, clean_data)
    time_series = dfClean[y]
    dates = dfClean[dates]

    time_series = time_series.set_index(dates)

    # run pth-order VAR
    model = VAR(time_series)
    results = model.fit(p)

    return results


# arguments: s = Sheet. y = var. dates = column name of dates for times series. p = number of lags in model
# returns: univariate (AR(p)) time series autoregression model.
def auto_reg(y, dates, p, s, clean_data="greedy"):

    v = np.copy(y)
    v = np.append(v, dates)

    # prepare data
    dfClean = s.cleanData(v, clean_data)
    time_series = dfClean[v]

    time_series = time_series.set_index(dates)
    model = s_ar.AR(time_series)
    results = model.fit(p)
    return results


# arguments: s = Sheeet. v = vector of column names
# returns PCA object
def principle_component_analysis(v, s, clean_data="greedy"):

    # prepare data and execute with library
    dfClean = s.cleanData(v, clean_data)
    data = dfClean[v]
    pca = PCA(data)
    return pca


# efficiently implemented choose function
def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


# super simple just finds the proportional probabilities
# relative to the lowest AIC value available.
def find_prob_given_AIC(min_aic, aic):
    return math.exp((min_aic - aic) / 2)


# find kth largest element in an unsorted array a not including 1s
def _kthLargest(a, k):
    aSorted = np.sort(a)
    n = len(a)
    for i in range(0, n):
        if aSorted[i] == 1:
            index = i
            break

    return aSorted[index - k + 1]


# returns 1d index of (x, y). we traverse down the first column and then down the second column etc
def _dfToVectorIndex(numRows, x, y):
    return (x * numRows) + y


# returns 2d index of x. we traverse down the first column and then down the second column etc
def _vectortoDFIndex(numRows, i):
    f = math.floor(i / numRows)
    r = i % numRows
    return [r, f]


# arguments: cols = vector of one or more column names. time = column of time index.
# min_lag and max_lag = domain of consideration
# returns: the optimal number of lags in a time series regression with the given variables
def find_optimal_lag_length(
    cols, time, s, multi, min_lag=1, max_lag=3, criterion="aic"
):

    # try:
    #     s = self.map_column_to_sheet(cols)
    #     multi = False
    # except:
    #     s = self.map_column_to_sheet(cols[0])
    #     multi = True

    df = s.df

    if multi:
        try:
            args_vector = np.append(cols, time)
            data = df[args_vector]
            data = data.set_index(time)
        except:
            data = df[cols]

        model = VAR(data)

    else:
        try:
            args_vector = np.array([cols, time])
            data = df[args_vector]
            data = data.set_index(time)
        except:
            data = df[cols]

        model = s_ar.AR(data)

    info_loss = np.zeros(max_lag - min_lag + 1)

    if criterion == "aic":
        for i in range(max_lag - min_lag + 1):
            fit = model.fit(i + min_lag)
            try:
                info_loss[i] = fit.aic
            except:
                utterance = "Error in Cholesky decomposition. (Positive definite specification not met.)"
                utterance = (
                    utterance + " Try decreasing the maximum number of lags considered."
                )
                return QueryResult(None, utterance)

    elif criterion == "bic":
        for i in range(max_lag - min_lag + 1):
            fit = model.fit(i + min_lag)
            try:
                info_loss[i] = fit.bic
            except:
                utterance = "Error in Cholesky decomposition. (Positive definite specification not met.)"
                utterance = (
                    utterance + " Try decreasing the maximum number of lags considered."
                )
                return QueryResult(None, utterance)

    else:
        print("ERROR: Criterion argument not supported.")
        return

    x = np.argsort(info_loss)
    optimal = x[0] + min_lag

    utterance = (
        "The optimal lag length according to the " + str(criterion) + " criterion is "
    )
    utterance = utterance + str(optimal) + "."

    return QueryResult(optimal, utterance)

    # candidates must be NumPy array!
    """def find_plausible_instruments(s, y, X, snitch, lengthX, short_list=-1, numToReturn=10, k=0, exog_regressors=-1, clean_data="greedy"):
        
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

        if f(X) is str:
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

    """
