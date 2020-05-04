from econlib import *
from random import *
import numpy as np
import pandas as pd
from scipy import stats

dfToy = pd.DataFrame(
    [[1, 2, 3, 19],
    [3, 5, -10, 6],
    [-4, -1, -11, 5]]
)

# return true iff x is a number
def isNumeric(x):
    if not type(x) is str:
        return True
    else: return False

# print initial summary statistics for the dataframe input.
# assumes row 0 has the names of the columns
def summarize(df):
    
    col = len(df.columns)
    for i in range(0, col):
        if (isNumeric(df.iloc[1, i])):
            print("mean of ", df.iloc[0, i], " is ", findmean(df.iloc[:, i]))
            print("standard deviation of ", df.iloc[0, i], " is ", findstd(df.iloc[:, i]))


print(dfToy)
summarize(dfToy)



