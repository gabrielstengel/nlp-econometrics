from random import *

import numpy as np
import pandas as pd
from scipy import stats


from athena_all.file_processing.excel import DataReader

dr = DataReader("./test_data/test_dataset.xlsx")
dfs = dr.get_all_sheets()
df = dfs[0]
df2 = dfs[1]

s = Sheet(dfs)

row = len(df)
col = len(df.columns)

v = findNumericColumns(df)

# grab one random numeric column for later
r = randint(0, col - 1)
while (v[r] != 1):
    r = randint(0, col - 1)

r = 4

# grab a second random numeric column for later
r2 = randint(0, col - 1)
while (v[r2] != 1 or r2 == r):
    r2 = randint(0, col - 1)

x = df.iloc[:, r]
y = df.iloc[:, r2]

# print(df['Y'])
# print(df.iloc[:,3])
# print(df.iloc[3])
# print(df.iloc[3,2])

# testing findmean()
def testMean():
    print("TESTING MEAN")
    if (np.mean(x) == findMean(x)):
        print(" correct")
    else:
        print("ERROR")

# testing findstd()
def testStd():
    print("TESTING STD")
    if (np.std(x) == findStd(x)):
        print(" correct")
    else:
        print("ERROR")

# testing findvar()
def testVar():
    print("TESTING VAR")
    if (np.var(x) == findVar(x)):
        print(" correct")
    else:
        print("ERROR")

# testing findmax()
def testMax():
    print("TESTING MAX")
    if (np.max(x) == findMax(x)):
        print(" correct")
    else:
        print("ERROR")

# testing findmin()
def testMin():
    print("TESTING MIN")
    if (np.min(x) == findMin(x)):
        print(" correct")
    else:
        print("ERROR")

# testing findmedian()
def testMedian():
    print("TESTING MEDIAN")
    if (np.median(x) == findMedian(x)):
        print(" correct")
    else: print("ERROR")

# tests all 3 using column name inputs
def testLargestCorrWithStrings():
    print("TESTING LARGEST CORR WITH A STRING INPUT")
    for i in range(0, col):
        if (v[i] == 0):
            continue
        else:
            t = df.columns[i]
            print(" column name: ", t)
            print("     output: ", findVariableName(df, largestCorr(s, 0, t)))

# call functions
print()
dfs = splitDataset(df, 3)
for i in range(0, 3):
    print(dfs[i])
    print()
print()
print("hello")
print()
print(df)
print()
print()
testMean()
print()
print()
testStd()
print()
testVar()
print()
print()
testMax()
print()
print()
testMin()
print()
print()
testMedian()
print()
print()
testLargestCorr()
print()
print()
testLargestCorrWithStrings()
print()
print()
print("printing correlation matrix for first df")
print(s.corr_matrices[0])
print()
x = [1,3,1,23,23,1,2]
y = [2,4,5,12,34,-2,0]

print()
print(df)
print()
print("here we go")
print(reg(df, 0, 2))
print()