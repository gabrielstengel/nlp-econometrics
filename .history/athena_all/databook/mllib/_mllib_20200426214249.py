import math
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
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
        X = s.get_numeric_columns()[0]

        # Setup our SVM
        svc = SVC(kernel="linear")

        cv_results = cross_validate(svc, X, y, cv=5)
        acc = cv_results["test_score"].mean()
        utterance = f"Using Support Vector Machines with 5-fold cross validation we found that the average test score was  {acc}."
        return QueryResult(utterance, utterance)

    def kmeans(self, k=5):
        numeric_cols = self.get_numeric_columns()
        kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(numeric_cols[0]))
        utterance = f"K-Means successfully converged after {kmeans.n_iter_} iterations. Only used numeric columns."
        utterance += f"\nHere are the cluster {k} centers. The values are presented in the column order:\n"
        utterance += f"\t\t{numeric_cols}"
        for clust in kmeans.cluster_centers_:
            utterance += f"\t\t{clust}\n"
        utterance += f"\nThe paramters were {kmeans.get_params}."

        return QueryResult(utterance, utterance)

    # ==================================================================================#
    # END OF ECONOMICS LIBRARY =========================================================#
    # ==================================================================================#

    def get_mllib_fmap(self):

        return [
            # ===================== Summary Statistic Methods ===========================#
            (
                ["classify", "predict", "use support vector machines"],
                "classify",
                self.classify,
            ),
            (["cluster", "kmeans", "k means"], "kmeans", self.kmeans),
        ]
        # ==============================================================================#
