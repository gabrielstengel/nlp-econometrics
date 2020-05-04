from athena_all.databook.queryresult import QueryResult


class UtilLibMixin:

    # ==================================================================================#
    # SHOWING THE DATA =================================================================#
    # ==================================================================================#

    def showCol(self, col):
        colvals = self.get_column(col)
        utterance = f"Column {col}: {colvals} "
        return QueryResult(colvals, utterance)

    def get_util_fmap(self):
        return [
            (["show column", "average", "avg"], "findMean", self.findMean),
        ]
