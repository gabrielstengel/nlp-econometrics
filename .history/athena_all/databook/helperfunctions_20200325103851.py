import random
import numpy as np
import pandas as pd
from scipy import stats
import numbers
import math
from athena_all.databook._sheetObj import *
import statsmodels.imputation.bayes_mi as smBayes


# takes a dataframe df. returns a vector of 0s and 1s.
# it's a 1 iff it corresponds to an index of a numeric column of df.
def findNumericColumns(df):

    col = len(df.columns)
    v = []
    for i in range(0, col):
        x = df.iloc[:,i]
        l = len(x) - 4
        j = 3
        while (j < l):
            xj = x[j]
            if (isinstance(xj, numbers.Number)):
                v.append(1)
                break
            if (pd.isna(xj)):
                j += 1
                continue
            else:
                v.append(0)
                break
    
    return v

# takes dataframe df and column name string. returns first index 
# with column name. doesn't check for multiple instances.
# returns -1 if column name not found
def findColumnIndexGivenName(df, string):
    v = (df.columns == string)
    j = -1
    for i in range(0, len(df.columns)):
        if v[i] == True:
            j = i
            break
    return j

# takes dataframe df and vector or scalar column indices x.
# returns name of corresponding column or columns 
def findVariableName(df, x):
    if np.isscalar(x):
        return df.columns[x]
    else:
        n = len(x)
        strVector = []
        for i in range(0, n):
            strVector.append(df.columns[x[i]])

        return strVector

# Function to insert row in the dataframe 
# (found at https://www.geeksforgeeks.org/insert-row-at-given-position-in-pandas-dataframe/) 
def insert_row(row_number, df, row_value): 

    start_upper = 0
    end_upper = row_number 
    start_lower = row_number 
    end_lower = df.shape[0] 
    upper_half = [*range(start_upper, end_upper, 1)] 
    lower_half = [*range(start_lower, end_lower, 1)] 
    lower_half = [x.__add__(1) for x in lower_half] 
    index_ = upper_half + lower_half 
    df.index = index_
    df.loc[row_number] = row_value
    df = df.sort_index()
    return df 

# find kth largest element in an unsorted array a not including 1s
def kthLargest(a, k):
    aSorted = np.sort(a)
    n = len(a)
    for i in range(0, n):
        if (aSorted[i] == 1):
            index = i
            break

    return aSorted[index - k + 1]

# v is a vector of column indices in df. this method returns a dataframe
# with incomplete rows removed. all columns are assumed to be of equal length. 
# all elements are assumed to be numbers or empty
def cleanData(df, vec, manner):

    if manner == False:
        return df

    if manner == "greedy":
    
        num_columns = len(vec)
        v = np.zeros(num_columns)

        if type(vec[0]) is np.str_:
            for i in range(num_columns):
                v[i] = findColumnIndexGivenName(df, vec[i])
        
        v = v.astype(int)

        num_rows = len(df.iloc[:,v[0]])

        for i in range(0, num_rows):
            for j in range(0, num_columns):

                var = df.iloc[:,v[j]][i]
                if (math.isnan(var)):
                    df = df.drop(i)
                    break

        return df

    # not yet tested. If it works it only gives the actual columns we're looking at
    # PROBABLY FUCKED BECAUSE THE COLUMN NAMES GET FUCKED
    # uses Gibbs sampling
    if manner == "bayesian_imputation":
        newDF = smBayes.BayesGaussMI(df[vec])
        return newDF.data

    
# randomly splits dataframe into different samples to increase generalizability. 
# returns array of dataframes dfs
def splitDataset(df, numberOfSplits):

    num_rows = len(df)
    x = np.arange(num_rows)

    np.random.seed(8)
    np.random.shuffle(x)

    numberInEach = round(num_rows / numberOfSplits) - 1

    dfs = []

    for i in range(0, numberOfSplits):
        v = np.zeros(numberInEach)
        
        for j in range(0, numberInEach):
            v[j] = x[j + numberInEach * i]
        
        dfs.append(df.iloc[v])

    return dfs

# returns 1d index of (x, y). we traverse down the first column and then down the second column etc
def dfToVectorIndex(numRows, x, y):
    return (x * numRows) + y

# returns 2d index of x. we traverse down the first column and then down the second column etc
def vectortoDFIndex(numRows, i):
    f = math.floor(i / numRows)
    r = i % numRows
    return [r, f]


# Adds new column to s.df with transformation. Adds column called col"_transformed"
# arg = additional argument. for add and multiply, "arg" is name of second col
def transformVariable(s, col, arg=0, power=False, ln=False, add=False, multiply=False, reprocess_sheet=False):

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

