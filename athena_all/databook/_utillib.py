import numpy as np
import matplotlib.pyplot as plt

from athena_all.databook.queryresult import QueryResult


class UtilLibMixin:

    # ==================================================================================#
    # SHOWING THE DATA =================================================================#
    # ====================================s==============================================#

    def showCol(self, col):
        colvals = np.array(list(self.get_column(col)))
        if len(colvals) > 10:
            colvals = f"[{colvals[0]}, {colvals[1]}, {colvals[2]}, {colvals[3]}, {colvals[4]}, ..., {colvals[5]}, {colvals[6]}, {colvals[7]}, {colvals[8]}, {colvals[9]}]"
        utterance = (
            f"These are some of the values in column {col.upper()}: \n{str(colvals)} "
        )
        return QueryResult(utterance, utterance)

    def listCols(self):
        col_names = [col.upper() for col in self.get_column_names()]
        utterance = f"The column names in this dataset: \n{col_names} "
        return QueryResult(utterance, utterance)

    def graph(self, col_names):
        if not type(col_names) == list:
            col_names = [col_names]

        col_data = [self.get_column(col) for col in col_names]
        for idx, col in enumerate(col_data):
            plt.plot(np.arange(len(col)), col, label=col_names[idx].upper())

        plt.xlabel("x - axis")
        plt.ylabel("y - axis")
        plt.legend()
        plt.show()
        return QueryResult("I made a graph for you!", "I made a graph for you!")

    def graphVs(self, col_x, col_y):
        plt.plot(self.get_column(col_x), self.get_column(col_y))
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.show()
        return QueryResult("I made a graph for you!", "I made a graph for you!")

    def greeting(self):
        utterance = "Great to meet you! Have fun using the app."
        return QueryResult(69, utterance)

    def help(self, functions=None):
        if functions is None or functions == "generic":
            utterance = ""
            utterance += "For help about a specific topic, just ask about it. (e.g. 'How do I run a regression?')"
            return QueryResult(utterance, utterance)

        q = _create_helpstring(functions)

        utterance = q
        denotation = q
        return QueryResult(denotation, utterance)

    def get_util_fmap(self):
        return [
            (["show column", "get column"], "showCol", self.showCol),
            (["greetings", "hello how are you"], "greeting", self.greeting),
            (["how does this work",], "help", self.help),
            (["graph"], "graph", self.graph),
            (["graphVs", "graph"], "graphVs", self.graphVs),
            (
                ["list all the columns", "all columns", "show columns",],
                "listCols",
                self.listCols,
            ),
        ]


def _create_helpstring(functions):
    q = "\n\tSorry, I couldn't understand that. But here's some advice based on what I could understand:\n\n"

    for idx, f in enumerate(set(functions)):
        q += "\t  (" + str(idx + 1) + ")\n"

        if f == "findMean":
            q = q + "\t\tFor calculating a mean, try:\n"
            q = q + "\t\twhat is the mean of _var_\n\n"

        if f == "findStd":
            q = q + "\t\tFor calculating a standard deviation, try:\n"
            q = q + "\t\twhat is the standard deviation of _var_\n\n"

        if f == "findVar":
            q = q + "\t\tFor calculating a variance, try:\n"
            q = q + "\t\twhat is the variance of _var_\n\n"

        if f == "findMax":
            q = q + "\t\tFor calculating a maximum, try:\n"
            q = q + "\t\twhat is the maximum value of _var_\n\n"

        if f == "findMin":
            q = q + "\t\tFor calculating a minimum, try:\n"
            q = q + "\t\twhat is the minimum value of _var_\n\n"

        if f == "findMedian":
            q = q + "\t\tFor calculating a median, try:\n"
            q = q + "\t\twhat is the median value of _var_\n\n"

        # ==================================================================================#
        # CORRELATION METHODS ==============================================================#
        # ==================================================================================#

        if f == "findCorr":
            q = q + "\t\tTo calculate a correlation between two variables, try:\n"
            q = q + "\t\twhat is the correlation between _var_1_ and _var_2_\n\n"

        if f == "largestCorr":
            q = q + "\t\tTo find the variable most correlated with a variable, try:\n"
            q = q + "\t\twhat variable is most correlated with _var_\n\n"

        if f == "largestCorrList":
            q = (
                q
                + "\t\tTo find the n variables most correlated with a variable, try:\n"
            )
            q = q + "\t\twhich n variables are most correlated with _var_\n\n"

        if f == "overallLargestCorrs":
            q = (
                q
                + "\t\tTo find the most significant correlations in the dataset, try:\n"
            )
            q = q + "\t\twhat are the most significant relationships in the dataset\n\n"

        # ==================================================================================#
        # SIMPLE REGRESSION METHODS ========================================================#
        # ==================================================================================#

        if f == "reg":
            q = q + "\t\tTo run a univariate regression, try:\n"
            q = q + "\t\tregress _dep_var_ on _ind_var_\n\n"

        if f == "multiReg":
            q = q + "\t\tTo run a regression of multiple variables, try:\n"
            q = q + "\t\tregress _dep_var_ on _ind_var_1_ and _ind_var_2_\n\n"

        if f == "fixedEffects":
            q = q + "\t\tTo run a fixed effects regression, try:\n"
            q = q + "\t\trun a fixed effects regression of _dep_var_ on _ind_var_"
            q = (
                q
                + " with _id_var_ as the id variable and _time_var_ as the time variable\n\n"
            )

        if f == "summarizeLogisticRegression":
            q = q + "\t\tTo run a logistic (discrete choice) regression, try:\n"
            q = q + "\t\tlogistic _binary_var_ on _ind_var_\n\n"

        if f == "logisticMarginalEffects":
            q = q + "\t\tTo calculate the marginal effects of a logistic model, try:\n"
            q = q + "\t\tmarginal effects _binary_var_ on _ind_var_\n\n"

        # ==================================================================================#
        # INSTRUMENTAL VARIABLE METHODS ====================================================#
        # ==================================================================================#

        if f == "ivRegress":
            q = q + "\t\tTo run a regression using a statistical instrument, try:\n"
            q = (
                q
                + "\t\trun an instrumental variable regression of _dep_var_ on _ind_var_"
            )
            q = (
                q
                + " using _instrument_var_1_ and _instrument_var_2_ as instruments\n\n"
            )

        if f == "homoskedasticJStatistic":
            q = (
                q
                + "\t\tTo test the exogeneity of an overidentified instrument model, try:\n"
            )
            q = (
                q
                + "\t\tin an instrumental variable regression of _dep_var_ on _ind_var_1_ and _ind_var_2_ "
            )
            q = (
                q
                + "using _instrument_var_1_ _instrument_var_2_ and _instrument_var_3_ as instruments , "
            )
            q = q + "run an overidentification test for exogeneity\n\n"

        if f == "test_weak_instruments":
            q = q + "\t\tTo test the relevance of a set of instruments, try:\n"
            q = q + "\t\ttest the strength of _instrument_var_1_ and _instrument_var_2_"
            q = q + " as instruments on the variable _ind_var_\n\n"

        if f == "find_instruments":
            q = q + "\t\tTo find plausibly exogenous instruments, try:\n"
            q = (
                q
                + "\t\tfind the best instruments among the candidates _instrument_var_1_ "
            )
            q = (
                q
                + "_instrument_var_2_ and _instrument_var_3_ if we run _dep_var_ on _ind_var_1_ "
            )
            q = (
                q
                + "and _ind_var_2_ with _exog_var_ known to be an exogenous instrument\n\n"
            )

        # ==================================================================================#
        # TIME SERIES METHODS ==============================================================#
        # ==================================================================================#

        if f == "print_a_bunch_of_AR_shit":
            q = q + "\t\tTo run univariate autoregression with p lags, try:\n"
            q = (
                q
                + "\t\trun an autoregression with p lags on _var_ using _time_var_ as the time variable\n\n"
            )

        if f == "augmented_dicky_fuller_test":
            q = (
                q
                + "\t\tTo test the stationarity of a time series variable via Dickey-Fuller, try:\n"
            )
            q = q + "\t\ttest the stationarity of _var_\n\n"

        if f == "summarize_VAR":
            q = (
                q
                + "\t\tTo run a vector autoregression on multiple variables using p lags, try:\n"
            )
            q = (
                q
                + "\t\trun a vector autoregression on _var_1_ _var_2_ and _var_3_ with p lags "
            )
            q = q + "using _time_var_ as the time variable\n\n"

        if f == "granger_causality_test":
            q = (
                q
                + "\t\tTo test whether a variable is Granger caused by other variables in a p-lag time series model, try:\n"
            )
            q = (
                q
                + "\t\tin a regression with p lags using _time_var_ as the time variable with _var_1_ _var_2_ and _var_3_ ,"
            )
            q = q + " is _var_i_ granger caused by the others\n\n"

        if f == "analyze_lags":
            q = q + "\t\tTo decide how many lags to use in a time series model, try:\n"
            q = (
                q
                + "\t\thow many lags should i use on _var_1_ _var_2_ and _var_3_ with "
            )
            q = q + "_time_var_ as the time variable\n\n"

        # ==================================================================================#
        # ADVANCED REGRESSION METHODS ======================================================#
        # ==================================================================================#

        if f == "print_PCA_wrapper":
            q = q + "\t\tTo run basic principle component analysis, try:\n"
            q = (
                q
                + "\t\trun principle component analysis on _var_1_ , _var_2_ , _var_3_ , and _var_4_ \n\n"
            )

        if f == "poisson_regression":
            q = q + "\t\tTo run a Poisson generalized linear model, try:\n"
            q = q + "\t\trun a poisson regression of _dep_var_ on _ind_var_ \n\n"

        if f == "markov_switching_regime_regression":
            q = q + "\t\tTo regress a quantity with n hidden Markov regimes, try:\n"
            q = q + "\t\trun a hidden markov model with _var_ with n regimes\n\n"

        if f == "choose_among_regression_models":
            q = (
                q
                + "\t\tTo choose from among a set of models by estimating out of sample generalizability "
            )
            q = q + "\t\tusing a penalty term system, try:\n"
            q = (
                q
                + "which of the following models has the best out of sample generalizability : "
            )
            q = (
                q
                + "_var_1_ , _var_2_ , and _var_3_ ; _var_4_ , _var_5_ , and _var_6_ ; "
            )
            q = q + "_var_7_ , _var_8_ , and _var_9_ \n\n"

    q = q + ""
    return q
