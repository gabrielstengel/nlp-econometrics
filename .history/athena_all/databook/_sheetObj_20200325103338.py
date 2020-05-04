
import numpy as np
import pandas as pd
import numbers
from athena_all.helperfunctions import *

class Sheet():
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




# empoyees = [('gabe',        34,                 'Sydney',       155),
#             ('None',        31,                 'Delhi',        177.5),
#             ('ben',         16,                  None,          8.1),
#             ('wielllter',   31,                 'Delhi',        167),
#             ('bags',        12,                 '122',          14.4),
#             ('beast',       'oops wrong data',  'Mumbai',       135),
#             (987,           "#nan",              None,          111),
#             (1,             None,               'Colombo',      111)
#             ]
 
# # Create a DataFrame object
# A = pd.DataFrame(empoyees, columns=['Name', 'Age', 'City', 'Marks'])

# print(A)
# print()
# sheet = Sheet(A)
# print(sheet.df)
# print()
# print('\n'.join([str(sheet.col_info[k]) for k in sheet.col_info]))

# '''
# dataTypeSeries = A.dtypes
 
# print('Data type of each column of Dataframe :')
# print(dataTypeSeries)

# A['Age'] = pd.to_numeric(A['Age'], errors="coerce")


# dataTypeSeries = A.dtypes
 
# print('Data type of each column of Dataframe :')
# print(dataTypeSeries)
# print(A['Age'])
# print()
# print((A['Name'].dtypes))
# print()

# dtypeCount =[A.iloc[:,i].apply(type).value_counts() for i in range(A.shape[1])]
# print(dtypeCount)
# print(len(dtypeCount))
# print('\n\n\n'.join([str(i) for i in dtypeCount]))
# print()
# print(isinstance(A['Age'][0], numbers.Number))

# [isinstance]

# import pdb; pdb.set_trace()
# '''