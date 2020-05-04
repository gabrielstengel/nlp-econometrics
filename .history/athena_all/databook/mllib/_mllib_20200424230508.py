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

        cv_results = cross_validate(svc, X, y, cv=5)
        utterance = f"The results are {cv_results}."
        return QueryResult(cv_results, utterance)

    # ==================================================================================#
    # END OF ECONOMICS LIBRARY =========================================================#
    # ==================================================================================#

    def get_econlib_fmap(self):

        return [
            # ===================== Summary Statistic Methods ===========================#
            (
                ["classify", "predict", "use support vector machines"],
                "classify",
                self.classify,
            ),
        ]
        # ==============================================================================#
