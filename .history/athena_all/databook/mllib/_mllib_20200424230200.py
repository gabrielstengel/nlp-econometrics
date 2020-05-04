import math
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

from athena_all.databook.queryresult import QueryResult

################################################################################


######## THE FUNCTIONS THAT NEED SOME TLC ARE: AR_with_moving_average


class MachineLearningMixin:

    # ==================================================================================#
    # SUMMARY STATISTIC METHODS ========================================================#
    # ==================================================================================#

    # argument: string column name col
    # returns basic results at implementing classification methods.
    def classify(self, col_y, crossval=True):

        # First, assure that col is a categorical variable.
        y = self.get_column(col_y)
        y = y.copy()
        y = y.astype("category")

        # Get the training data
        s = self.map_column_to_sheet(col_y)
        X = s.get_numeric_columns()

        # Setup our SVM
        svc = SVC(kernel="linear")

        mean = np.mean(self.get_column(col))
        utterance = "The mean of " + str(col) + " is " + str(mean) + "."
        return QueryResult(mean, utterance)

    # ==================================================================================#
    # END OF ECONOMICS LIBRARY =========================================================#
    # ==================================================================================#

    def get_econlib_fmap(self):

        return [
            # ===================== Summary Statistic Methods ===========================#
            (["mean", "average", "avg"], "findMean", self.findMean),
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
        ]
        # ==============================================================================#
