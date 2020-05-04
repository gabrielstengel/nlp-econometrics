from athena_all.databook.queryresult import QueryResult

import numpy as np


class UtilLibMixin:

    # ==================================================================================#
    # SHOWING THE DATA =================================================================#
    # ====================================s==============================================#

    def showCol(self, col):
        colvals = np.array(list(self.get_column(col)))
        if len(colvals) > 10:
            colvals = f"[{colvals[0]}, {colvals[1]}, {colvals[2]}, {colvals[3]}, {colvals[4]}, ..., {colvals[5]}, {colvals[6]}, {colvals[7]}, {colvals[8]}, {colvals[9]}]"
        utterance = f"These are some of the values in column {col}: \n{str(colvals)} "
        return QueryResult(utterance, utterance)

    def listCols(self):
        col_names = self.get_column_names()
        utterance = f"The column names in this dataset: {col_names} "
        return QueryResult(utterance, utterance)

    def greeting(self):
        utterance = "Great to meet you! Have fun using the app."
        return QueryResult(69, utterance)

    def help(self, functions=None):
        if functions is None:
            utterance = "Sorry, I didn't understand that."
            return QueryResult(utterance, utterance)

        """ FOR WILLETT TO FILL OUT """
        ##
        ##
        ##

        utterance = ""
        denotation = ""
        return QueryResult(denotation, utterance)

    def get_util_fmap(self):
        return [
            (["show column", "get column"], "showCol", self.showCol),
            (["greetings", "hello how are you"], "greeting", self.greeting),
            (["how does this work",], "help", self.help),
            (
                [
                    "show me all the columns",
                    "list all the columns",
                    "all columns",
                    "show columns",
                ],
                "listCols",
                self.listCols,
            ),
        ]
