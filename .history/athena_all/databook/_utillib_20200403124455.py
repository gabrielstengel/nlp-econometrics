from athena_all.databook.queryresult import QueryResult


class UtilLibMixin:

    # ==================================================================================#
    # SHOWING THE DATA =================================================================#
    # ==================================================================================#

    def showCol(self, col):
        colvals = self.get_column(col)
        import pdb

        pdb.set_trace()
        utterance = f"Column {col}: {colvals} "
        return QueryResult(colvals, utterance)

    def get_util_fmap(self):
        return [
            (["show column", "get column"], "showCol", self.showCol),
        ]

