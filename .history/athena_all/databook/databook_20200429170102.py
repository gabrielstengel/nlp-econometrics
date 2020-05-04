from athena_all.databook.sheet import Sheet
from athena_all.databook.econlib import EconLibMixin
from athena_all.databook._utillib import UtilLibMixin
from athena_all.databook.mllib import MachineLearningMixin

from athena_all.databook.queryresult import QueryResult


class DataBook(EconLibMixin, UtilLibMixin, MachineLearningMixin):
    def __init__(self):
        """ Stores a number of sheets representing different excel files. Also inherits
        a number of methods for doing calculations on those sheets. """

        self.sheets = []
        self.results = []
        self.functions = (
            self.get_econlib_fmap() + self.get_util_fmap() + self.get_mllib_fmap()
        )

    def add_df(self, df, sheet_name=None):
        """ Adds a dataframe to the DataBook. """
        self.sheets.append(Sheet(df))

    def add_dfs(self, dfs):
        """ Adds a dataframe to the DataBook. """
        assert len(dfs) > 0
        for df in dfs:
            self.add_df(df)

    def map_column_to_sheet(self, colname):
        """ Given a column name, return the sheet that the column belongs to """

        # Must have at least one sheet. Column name must be lowercase.
        assert len(self.sheets) > 0, "No sheets in the DataBook."

        for sheet in self.sheets:
            if colname.lower() in sheet.col_info:
                return sheet

        return None

    def get_column(self, colname):
        """ Given a column name return the column. """

        return self.map_column_to_sheet(colname).df[colname]

    def get_function_names(self):
        return [name for _, name, _ in self.functions]

    def get_function_synonyms(self):
        return [synonyms for synonyms, _, _ in self.functions]

    def get_function_lambdas(self):
        return [operation for _, _, operation in self.functions]

    def get_names_to_lambdas(self):
        return {kv[1]: kv[2] for kv in self.functions}

    def get_numeric_columns(self):
        cols = []
        for s in self.sheets:
            cols += s.get_numeric_columns()
        return cols

    def execute_func(self, func_name, args, denotation_only=True):
        try:
            result = self.get_names_to_lambdas()[func_name](*args)
        except:
            utter = "Oops, I had an error in my maths. Are you sure all your arguments are amenable to the type of function you're trying to run?"
            result = QueryResult(utter, utter)
        if denotation_only:
            return result.get_denotation()
        return result

    def get_column_names(self):
        cols = []
        for s in self.sheets:
            cols += s.get_col_names()
        return [col.lower() for col in cols]
