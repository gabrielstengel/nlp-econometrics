import athena_all.econlib as econlib

# Basic Arithmatic Operations for numbers.
def get_arithmatic_ops():
    return {
        '~': lambda x: -x,
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '/': lambda x, y: x / y,
        '*': lambda x, y: x * y,
        'avg': lambda x, y: (x + y) / 2 ,
        '^2': lambda x: x ** 2,
        '^3': lambda x: x ** 3,
        '^1/2': lambda x: x ** (1/2),
    }

# Column operations.
def get_column_ops(df):
    return {
        'applyFunctionManyTimes': lambda f, args: [f(arg) for arg in args],
        'getAllColumns': lambda x: [df[col] for col in df.columns],
        'getCol': lambda col: df[col],
    }


# Econometric Library of Operations.
def get_econlib_ops():
    return {
        'findMean': lambda x: econlib.findMean(x),
        'findStd': lambda x: econlib.findStd(x),
        'findVar': lambda x: econlib.findVar(x),
        'findMax': lambda x: econlib.findMax(x),
        'findMin': lambda x: econlib.findMin(x),
        'findCorr': lambda sheet, x, y: econlib.findCorr(sheet, x, y),
        'largestCorr': lambda sheet, col: econlib.largestCorr(sheet, col),
        'largestCorrList': lambda sheet, col, num_return: econlib.largestCorrList(sheet, col, num_return),
        'overallLargestCorrs': lambda sheet, num_return: econlib.overallLargestCorrs(sheet, num_return),
        'reg': lambda sheet, dependent_col, independent_col: econlib.reg(sheet, )
        }