
import numpy as np
import pandas as pd
import numbers
from athena_all.databook.helperfunctions import *
import random
import numpy as np
import pandas as pd
from scipy import stats
import numbers
import math
from athena_all.databook._sheetObj import *
import statsmodels.imputation.bayes_mi as smBayes

class Sheet(object):
    def __init__(self, df: pd.DataFrame):
        ''' 
        Method to initialize a sheet. A sheet contains the information for one sheet from an
        excel file, and makes up only one dataframe, as well as info about the data and how 
        it is stored. Takes a dataframe as an argument

        Possible options for col_info:
            1. type
            2. percentage_numeric
            3. num_not_numeric
            4. num_nan
            5. synonyms
            6. calculated_*anything*
        '''

        self.df, self.col_info = self._process_df(df)
        self.corr_matrix = df.corr(method = 'pearson')


    def _process_df(self, df, numeric_minimum = 4):
        col_types = {}
        orig_df = df.copy(deep=True)

        # Iterate through each column and figure out what its data is
        # Change numeric columns to all be the same number type and get rid of none/nan
        for col in orig_df:
            num_numerics = np.sum([isinstance(orig_df[col][i], numbers.Number) for i in range(len(orig_df[col]))])

            if num_numerics > numeric_minimum:
                df[col] = pd.to_numeric(orig_df[col], errors = 'coerce')
                col_types[col] = {'type':                    'numeric',
                                  'percentage_numeric':      100 * num_numerics / len(orig_df[col]),
                                  'num_not_numeric':         len(orig_df[col]) - num_numerics,
                                  'num_nan':                 orig_df[col].isna().sum()}
            else:
                col_types[col] = {'type':       'string',
                                  'num_nan':    orig_df[col].isna().sum()}

        return df, col_types




    # takes dataframe df and column name string. returns first index 
    # with column name. doesn't check for multiple instances.
    # returns -1 if column name not found
    def findColumnIndexGivenName(self, string):
        v = (self.df.columns == string)
        j = -1
        for i in range(0, len(self.df.columns)):
            if v[i] == True:
                j = i
                break
        return j

    # takes dataframe df and vector or scalar column indices x.
    # returns name of corresponding column or columns 
    def findVariableName(self, x):
        if np.isscalar(x):
            return self.df.columns[x]
        else:
            n = len(x)
            strVector = []
            for i in range(0, n):
                strVector.append(self.df.columns[x[i]])

            return strVector


    # v is a vector of column indices in df. this method returns a dataframe
    # with incomplete rows removed. all columns are assumed to be of equal length. 
    # all elements are assumed to be numbers or empty
    def cleanData(self, vec, manner):

        if manner == False:
            return self.df

        if manner == "greedy":
        
            num_columns = len(vec)
            v = np.zeros(num_columns)

            if type(vec[0]) is np.str_:
                for i in range(num_columns):
                    v[i] = findColumnIndexGivenName(self.df, vec[i])
            
            v = v.astype(int)

            num_rows = len(self.df.iloc[:,v[0]])

            for i in range(0, num_rows):
                for j in range(0, num_columns):

                    var = self.df.iloc[:,v[j]][i]
                    if (math.isnan(var)):
                        self.df = self.df.drop(i)
                        break

            return self.df

        # not yet tested. If it works it only gives the actual columns we're looking at
        # PROBABLY FUCKED BECAUSE THE COLUMN NAMES GET FUCKED
        # uses Gibbs sampling
        if manner == "bayesian_imputation":
            newDF = smBayes.BayesGaussMI(df[vec])
            return newDF.data

        
    # randomly splits dataframe into different samples to increase generalizability. 
    # returns array of dataframes dfs
    def splitDataset(self, numberOfSplits):

        num_rows = len(self.df)
        x = np.arange(num_rows)

        np.random.seed(8)
        np.random.shuffle(x)

        numberInEach = round(num_rows / numberOfSplits) - 1

        dfs = []

        for i in range(0, numberOfSplits):
            v = np.zeros(numberInEach)
            
            for j in range(0, numberInEach):
                v[j] = x[j + numberInEach * i]
            
            dfs.append(self.df.iloc[v])

        return dfs


    # Adds new column to s.df with transformation. Adds column called col"_transformed"
    # arg = additional argument. for add and multiply, "arg" is name of second col
    def transformVariable(self, col, arg=0, power=False, ln=False, add=False, multiply=False, reprocess_sheet=False):

        newColName = col + "_transformed"
        i = 1

        while (True):
            i += 1
            if not newColName + str(i) in s.df:
                newColName = newColName + str(i)
                break

        if power:
            s.df[newColName] = np.power((s.df[col]), arg)
        if ln:
            s.df[newColName] = np.log(s.df[col])
        if add:
            s.df[newColName] = s.df.loc[:,[col,arg]].sum(axis=1)
        if multiply:
            s.df[newColName] = s.df[col]*s.df[arg]
        if reprocess_sheet:
            s = Sheet(s.df)

