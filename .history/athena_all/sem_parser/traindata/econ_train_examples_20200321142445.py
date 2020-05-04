from athena_all.sem_parser.grammar.example import Example

import pandas as pd


def train_example_matrix():
    return pd.DataFrame([[1, 0, 2019],
                         [1, 2, 2020],
                         [0, 2, 2021]],
                         columns=['ENTERQ', 'AGE', 'SPEND'])


train_examples =  [
    # --------------------------------------------------------------------------------------------------------------- #
    # findMean: proper grammar
    Example(input="Find the mean of ENTERQ", semantics=('findMean', ('get_col', 'ENTERQ')), to_lower=True, denotation=2/3),
    Example(input="What is the mean of AGE", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),
    Example(input="Find the average of ENTERQ", semantics=('findMean', ('get_col', 'ENTERQ')), to_lower=True, denotation=2/3),
    Example(input="What is the average of AGE", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),
    Example(input="Find the avg of ENTERQ", semantics=('findMean', ('get_col', 'ENTERQ')), to_lower=True, denotation=2/3),
    Example(input="What is the avg of AGE", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),

    # findMean: improper grammar
    Example(input="Show mean of ENTERQ", semantics=('findMean', ('get_col', 'ENTERQ')), to_lower=True, denotation=2/3),
    Example(input="mean of AGE", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),
    Example(input="Find the average of ENTERQ", semantics=('findMean', ('get_col', 'ENTERQ')), to_lower=True, denotation=2/3),
    Example(input="Whats the average of AGE", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),
    Example(input="What's the mean of AGE", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),
    Example(input="average of ENTERQ", semantics=('findMean', ('get_col', 'ENTERQ')), to_lower=True, denotation=2/3),
    Example(input="average AGE", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),
    Example(input="AGE's mean", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),
    Example(input="AGEs mean", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),
    Example(input="AGE's average", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),
    Example(input="Show AGE's mean", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),
    Example(input="Find AGEs mean", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),
    Example(input="What is AGE's average", semantics=('findMean', ('get_col', 'AGE')), to_lower=True, denotation=4/3),

    # --------------------------------------------------------------------------------------------------------------- #
    # findStd: proper gramamr
    Example(input="Find the std of ENTERQ", semantics=('findStd', ('get_col', 'ENTERQ')), to_lower=True, denotation=0.4714045207910317),
    Example(input="What is the standard deviation of AGE", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),
    Example(input="Find the standard deviation of ENTERQ", semantics=('findStd', ('get_col', 'ENTERQ')), to_lower=True, denotation=0.4714045207910317),
    Example(input="What is the std of AGE", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),
    Example(input="Find the std of ENTERQ", semantics=('findStd', ('get_col', 'ENTERQ')), to_lower=True, denotation=0.4714045207910317),
    Example(input="What is the standard dev of AGE", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),

    # findStd: improper grammar
    Example(input="Show standard deviation of ENTERQ", semantics=('findStd', ('get_col', 'ENTERQ')), to_lower=True, denotation=0.4714045207910317),
    Example(input="std of AGE", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),
    Example(input="Find the std of ENTERQ", semantics=('findStd', ('get_col', 'ENTERQ')), to_lower=True, denotation=0.4714045207910317),
    Example(input="Whats the std deviation of AGE", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),
    Example(input="What's the stddev of AGE", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),
    Example(input="standard deviation of ENTERQ", semantics=('findStd', ('get_col', 'ENTERQ')), to_lower=True, denotation=0.4714045207910317),
    Example(input="standard deviation AGE", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),
    Example(input="AGE's stddev", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),
    Example(input="AGEs stddev", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),
    Example(input="AGE's standard deviation", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),
    Example(input="Show AGE's stddev", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),
    Example(input="Find AGEs stddev", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),
    Example(input="What is AGE's standard deviation", semantics=('findStd', ('get_col', 'AGE')), to_lower=True, denotation=0.9428090415820634),

    # --------------------------------------------------------------------------------------------------------------- #
    # findVar: proper grammar
    Example(input="Find the var of ENTERQ", semantics=('findVar', ('get_col', 'ENTERQ')), to_lower=True, denotation=0.22222222222222224),
    Example(input="What is the variance of AGE", semantics=('findVar', ('get_col', 'AGE')), to_lower=True, denotation=0.888888888888889),
    Example(input="Find the spread of ENTERQ", semantics=('findVar', ('get_col', 'ENTERQ')), to_lower=True, denotation=0.22222222222222224),
    Example(input="What is the var of AGE", semantics=('findVar', ('get_col', 'AGE')), to_lower=True, denotation=0.888888888888889),
    Example(input="Find the variance of ENTERQ", semantics=('findVar', ('get_col', 'ENTERQ')), to_lower=True, denotation=0.22222222222222224),
    Example(input="What is the var of AGE", semantics=('findVar', ('get_col', 'AGE')), to_lower=True, denotation=0.888888888888889),

    # --------------------------------------------------------------------------------------------------------------- #
    # findMax: proper grammar
    Example(input="Find the max of ENTERQ", semantics=('findMax', ('get_col', 'ENTERQ')), to_lower=True, denotation=1),
    Example(input="What is the max of AGE", semantics=('findMax', ('get_col', 'AGE')), to_lower=True, denotation=2),
    Example(input="Find the max value in ENTERQ", semantics=('findMax', ('get_col', 'ENTERQ')), to_lower=True, denotation=1),
    Example(input="What is the biggest value in AGE", semantics=('findMax', ('get_col', 'AGE')), to_lower=True, denotation=2),
    Example(input="Find the largest ENTERQ value", semantics=('findMax', ('get_col', 'ENTERQ')), to_lower=True, denotation=1),
    Example(input="What is the biggest entry in AGE", semantics=('findMax', ('get_col', 'AGE')), to_lower=True, denotation=2),

    # --------------------------------------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------------------------------------- #
]




print(train_example_matrix())