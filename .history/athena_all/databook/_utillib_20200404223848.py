from athena_all.databook.queryresult import QueryResult


class UtilLibMixin:

    # ==================================================================================#
    # SHOWING THE DATA =================================================================#
    # ====================================s==============================================#

    def showCol(self, col):
        colvals = list(self.get_column(col))
        utterance = f"Column {col}: {colvals} "
        return QueryResult(colvals, utterance)

    def greeting(self):
        utterance = "Great to meet you! Have fun using the app."
        return QueryResult(69, utterance)
    
    def help(self, subject):s

    def get_util_fmap(self):
        return [
            (["show column", "get column"], "showCol", self.showCol),
            (["greetings", "hello how are you"], "greeting", self.greeting),
        ]

