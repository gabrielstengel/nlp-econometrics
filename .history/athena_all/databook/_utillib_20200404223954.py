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

    def help(self, subject):
        utterance = "Mhm, I'll try and help as best I can. Would you like a rundown of how everything works, or are you curious about a specific subuject?"
        return QueryResult(6969, utterance)

    def get_util_fmap(self):
        return [
            (["show column", "get column"], "showCol", self.showCol),
            (["greetings", "hello how are you"], "greeting", self.greeting),
        ]

