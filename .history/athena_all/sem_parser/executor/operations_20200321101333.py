import athena_all.econlib as econlib



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


def get_column_ops(df):
    return {
        'applyFunctionManyTimes': lambda f, args: [f(arg) for arg in args],
        'getAllColumns': lambda x: [df[col] for col in df.columns],
        'getCol': lambda col: df[col],
    }






def get_econlib_ops():
    return {
        'findMean': lambda x: econlib.findMean(x),
        'findStd': lambda x: econlib.findStd(x),
        'findVar': lambda x: econlib.findVar(x),
        'findMax': lambda x: econlib.findMax(x),
        'findMin': lambda x: econlib.findMin(x)
        }