import math
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

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
        svc_cv_results = cross_validate(svc, X, y, cv=5)
        svc_acc = round(svc_cv_results["test_score"].mean() * 100, 2)

        # Setup Naive Bayes
        gnb = GaussianNB()
        gnb_cv_results = cross_validate(gnb, X, y, cv=5)
        gnb_acc = round(gnb_cv_results["test_score"].mean() * 100, 2)

        # Set up Random Forest
        rdf = RandomForestClassifier(random_state=0)
        rdf_cv_results = cross_validate(rdf, X, y, cv=5)
        rdf_acc = round(rdf_cv_results["test_score"].mean() * 100, 2)

        def _name_best_method():
            if svc_acc > gnb_acc:
                if svc_acc > rdf_acc:
                    return "Support Vector Machines"
                return "Random Forst"
            if gnb_acc > rdf_acc:
                return "Gaussian Naive Bayes"
            return "Random Forest"

        utterance = f"Below are the best performing classification algorithms. All accuracies are the mean result using 5-fold cross validation.\n"
        utterance += f"\tSupport Vector Machines: {svc_acc}\n"
        utterance += f"\tGaussian Naive Bayes: {gnb_acc}\n"
        utterance += f"\tRandom Forest: {rdf_acc}\n"
        utterance += f"We recomend using {_name_best_method()}. You can search for optimized hyperparameters using Athena as well."

        return QueryResult(utterance, utterance)

    def kmeans(self, k=5):
        numeric_cols = self.get_numeric_columns()
        kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(numeric_cols[0]))
        utterance = f"K-Means successfully converged after {kmeans.n_iter_} iterations. Only used numeric columns."
        utterance += f"\nHere are the cluster's {k} centers.\n"
        # utterance += f"\nHere are the cluster's {k} centers. The values are presented in the column order:\n"
        # utterance += f"\t\t{numeric_cols[1]}\n\n"
        for clust in kmeans.cluster_centers_:
            utterance += f"\t\t{[(round(val , 2)) for val in clust]}\n"

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
