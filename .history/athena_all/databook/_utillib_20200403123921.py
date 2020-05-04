from athena_all.databook.queryresult import QueryResult


class UtilLibMixin:

    # ==================================================================================#
    # SHOWING THE DATA =================================================================#
    # ==================================================================================#

    def showCol(self, col):
        colvals = self.get_column(col)
        return QueryResult(self.get_column(col),)
