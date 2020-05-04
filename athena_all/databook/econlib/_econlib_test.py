

from random import *

import numpy as np
import pandas as pd
from scipy import stats

from athena_all.file_processing.excel import DataReader

from athena_all.databook.sheet import Sheet
from athena_all.databook.databook import DataBook
from athena_all.databook.econlib import EconLibMixin


def print_line_break():
    for i in range(150):
        print("-", end="")
    print()



df = pd.DataFrame(
            [
                [
                    0,
                    0.59542555,
                    1,
                    0.57453056,
                    0.42934533,
                    0.93628182,
                    0.28526646,
                    0.2809299,
                    0.46671189,
                    0.72810118,
                    0.44030829,
                    0.96556485,
                    0.25916358,
                    1990,
                    2000,
                ],
                [
                    0,
                    0.61189131,
                    1,
                    0.04215727,
                    0.16399994,
                    0.05297215,
                    0.61953736,
                    0.52455045,
                    0.38079513,
                    0.86994707,
                    0.48356991,
                    0.38525283,
                    0.38243882,
                    1991,
                    2001,
                ],
                [
                    1,
                    0.93836554,
                    1,
                    0.69864825,
                    0.35213344,
                    0.59797455,
                    0.46134395,
                    0.054820202,
                    0.54327067,
                    0.70455268,
                    0.94702998,
                    0.52578236,
                    0.25243269,
                    1992,
                    2002,
                ],
                [
                    1,
                    0.43298268,
                    2,
                    0.27189097,
                    0.25729616,
                    0.36025735,
                    0.99944299,
                    0.58684202,
                    0.18428937,
                    0.9908807,
                    0.83407292,
                    0.86499515,
                    0.62944832,
                    1990,
                    2003,
                ],
                [
                    0,
                    0.04992074,
                    2,
                    0.82981333,
                    0.40672948,
                    0.21775362,
                    0.03243588,
                    0.88546157,
                    0.70340887,
                    0.59929183,
                    0.81263212,
                    0.19264746,
                    0.53919369,
                    1991,
                    2004,
                ],
                [
                    1,
                    0.20321437,
                    2,
                    0.0038075,
                    0.06510299,
                    0.96703176,
                    0.03240965,
                    0.54338895,
                    0.78314553,
                    0.85142279,
                    0.63651936,
                    0.48405215,
                    0.34185377,
                    1992,
                    2005,
                ],
                [
                    1,
                    0.67269108,
                    3,
                    0.51324964,
                    0.66779944,
                    0.65241317,
                    0.45445467,
                    0.846719,
                    0.58515694,
                    0.50807893,
                    0.04103975,
                    0.78657887,
                    0.49613462,
                    1990,
                    2006,
                ],
                [
                    1,
                    0.17131459,
                    3,
                    0.27865972,
                    0.71309775,
                    0.39422096,
                    0.14730283,
                    0.29544907,
                    0.12249358,
                    0.7590087,
                    0.80808535,
                    0.51282552,
                    0.93583393,
                    1991,
                    2007,
                ],
                [
                    1,
                    0.2660876,
                    3,
                    0.2801914,
                    0.89741106,
                    0.32327359,
                    0.27661438,
                    0.42924367,
                    0.38075493,
                    0.30701931,
                    0.70821542,
                    0.06626643,
                    0.71189485,
                    1992,
                    2008,
                ],
                [
                    0,
                    0.58024302,
                    4,
                    0.9348548,
                    0.70053465,
                    0.65790739,
                    0.77234365,
                    0.74697521,
                    0.39609364,
                    0.40388383,
                    0.54745616,
                    0.39264645,
                    0.24909293,
                    1990,
                    2009,
                ],
                [
                    1,
                    0.69867599,
                    4,
                    0.05007366,
                    0.45314885,
                    0.08157756,
                    0.68526291,
                    0.79869134,
                    0.77266248,
                    0.95665045,
                    0.32346161,
                    0.52262228,
                    0.96717552,
                    1991,
                    2010,
                ],
                [
                    0,
                    0.95127682,
                    4,
                    0.83831714,
                    0.98063061,
                    0.87853651,
                    0.08189685,
                    0.74680202,
                    0.7034782,
                    0.69459252,
                    0.03451852,
                    0.72770871,
                    0.85641232,
                    1992,
                    2011,
                ],
                [
                    0,
                    0.59872574,
                    5,
                    0.95031834,
                    0.63737451,
                    0.48484222,
                    0.53107266,
                    0.9249519,
                    0.26123892,
                    0.29084291,
                    0.73399499,
                    0.20534203,
                    0.07935591,
                    1990,
                    2012,
                ],
                [
                    0,
                    0.73638132,
                    5,
                    0.85611363,
                    0.74338016,
                    0.13075587,
                    0.26471049,
                    0.77041611,
                    0.3807917,
                    0.40610467,
                    0.46459439,
                    0.55694167,
                    0.55807716,
                    1991,
                    2013,
                ],
                [
                    0,
                    0.44916671,
                    5,
                    0.39273352,
                    0.14518805,
                    0.04278213,
                    0.28839647,
                    0.86804568,
                    0.27519021,
                    0.50247241,
                    0.30593673,
                    0.91597386,
                    0.51015697,
                    1992,
                    2014,
                ],
                [
                    1,
                    0.86606795,
                    6,
                    0.04756454,
                    0.54001659,
                    0.05116116,
                    0.20060841,
                    0.32426698,
                    0.32719354,
                    0.47876589,
                    0.65582939,
                    0.33074798,
                    0.03539574,
                    1990,
                    2015,
                ],
                [
                    1,
                    0.87308389,
                    6,
                    0.44709819,
                    0.74205641,
                    0.97884679,
                    0.62707191,
                    0.67062549,
                    0.07589355,
                    0.74740025,
                    0.17595032,
                    0.20191205,
                    0.17728823,
                    1991,
                    2016,
                ],
                [
                    0,
                    0.56842842,
                    6,
                    0.36726035,
                    0.78221789,
                    0.16303869,
                    0.81146834,
                    0.35431541,
                    0.090146,
                    0.65147732,
                    0.49358801,
                    0.24689245,
                    0.12729951,
                    1992,
                    2017,
                ],
            ],
            columns=[
                "DUMMY",
                "GDP",
                "PERSON_ID",
                "ENTERQ",
                "TANK",
                "BEAST",
                "ANIMAL",
                "AGE",
                "STINGER",
                "BEDWARDS",
                "EASY",
                "AUTO",
                "MATIC",
                "PANEL_YEAR",
                "YEAR",
            ],
        )


db = DataBook()
db.add_df(df)
s = db.map_column_to_sheet("MATIC")
print(s.df)
print()
print(s.corr_matrix)
print()

dep_vars_1 = ["stinger", "matic", "beast"]
dep_vars_2 = ["bedwards", "tank", "matic"]
dep_vars_3 = ["gdp", "easy", "auto", "matic"]
dep_vars_4 = ["easy", "matic", "enterq", "year", "animal"]

instruments_1 = ["gdp", "easy", "auto", "dummy"]
instruments_2 = ["stinger", "matic", "beast", "auto"]
instruments_3 = ["stinger", "person_id", "dummy", "year", "age"]
instruments_4 = ["gdp", "panel_year", "dummy", "beast", "tank", "stinger"]

print()
print()
# print(db.print_PCA_wrapper(["stinger", "bedwards"]).get_utterance())
# print()
# print()
# print(db.print_PCA_wrapper(["b", "c", "d"]).get_utterance())
# print()
print(db.print_a_bunch_of_AR_shit("stinger", "year", 3).get_utterance())
print()
print()


'''
iv_fit_1 = db.ivRegress("bedwards", dep_vars_1, instruments_1).denotation
iv_fit_2 = db.ivRegress("gdp", dep_vars_2, instruments_2).denotation
iv_fit_3 = db.ivRegress("beast", dep_vars_3, instruments_3).denotation
iv_fit_4 = db.ivRegress("age", dep_vars_4, instruments_4).denotation

print(iv_fit_1)
print()
print(iv_fit_2)
print()
print(iv_fit_3)
print()
print(iv_fit_4)
print()

print_line_break()
print(db.test_weak_instruments("bedwards", dep_vars_1).denotation)
print()
print(db.test_weak_instruments("stinger", dep_vars_2).denotation)
print()
print(db.test_weak_instruments("enterq", dep_vars_3).denotation)
print()
print_line_break()
print(db.find_instruments("tank", "enterq", "bedwards", dep_vars_1).denotation)
print()
print(db.find_instruments("beast", "enterq", "stinger", dep_vars_2).denotation)
print()
print(db.find_instruments("bedwards", "enterq", "stinger", dep_vars_3).denotation)
print()
print_line_break()
print(db.logisticMarginalEffects("dummy", dep_vars_1).get_denotation())
print()
print(db.logisticMarginalEffects("dummy", dep_vars_2).get_denotation())
print()
print(db.logisticMarginalEffects("dummy", dep_vars_3).get_denotation())
print()
print(db.logisticMarginalEffects("dummy", dep_vars_4).get_denotation())
print()
print_line_break()
print(db.print_a_bunch_of_AR_shit("stinger", "year", 4).denotation)
print()
print(db.print_a_bunch_of_AR_shit("beast", "year", 7).denotation)
print()
print(db.print_a_bunch_of_AR_shit("auto", "year", 3).denotation)
print()
print(db.print_a_bunch_of_AR_shit("matic", "year", 11).denotation)
print()
print_line_break()
print(db.summarize_VAR(dep_vars_1, "year", 3).denotation)
print()
print(db.summarize_VAR(dep_vars_2, "year", 2).denotation)
print()
print(db.summarize_VAR(dep_vars_3, "year", 2).denotation)
print()
print(db.summarize_VAR(dep_vars_4, "year", 4).denotation)
print()
print_line_break()
dep_vars_1 = ["stinger", "matic", "beast"]
print(db.granger_causality_test(dep_vars_1, "year", 3, "stinger").denotation)
print()
print(db.granger_causality_test(dep_vars_2, "year", 3, "bedwards").denotation)
print()
print(db.granger_causality_test(dep_vars_3, "year", 3, "gdp").denotation)
print()
print_line_break()
print(db.print_PCA_wrapper(["stinger", "beast"]).denotation)
print()
print(db.print_PCA_wrapper(["auto", "matic"]).denotation)
print()
print(db.print_PCA_wrapper(["dummy", "enterq"]).denotation)
print()


print_line_break()
print("                                                         #### --- INITIALIZING OBJECTS --- ####")
print_line_break()


dr = DataReader("debug_everything.xlsx")
dfs = dr.get_all_sheets()
df_before = dfs[0]

db = DataBook()
db.add_df(df_before)
s = db.map_column_to_sheet("a")
print_line_break()

print("                                                         #### --- PRINT OUT INITIAL INFO --- ####")
print_line_break()
print(s.df)
print()
print(s.corr_matrix)
print()
print_line_break()


print("                                                         #### --- CALL SUMMARY STATISTIC METHODS --- ####")
print_line_break()
print()
print(db.findMean("id").get_denotation())
print(db.findMean("id").get_utterance())
print()
print_line_break()
print()
print(db.findStd("a").get_denotation())
print(db.findStd("a").get_utterance())
print()
print_line_break()
print()
print(db.findVar("a").get_denotation())
print(db.findVar("a").get_utterance())
print()
print_line_break()
print()
print(db.findMedian("a").get_denotation())
print(db.findMedian("a").get_utterance())
print()
print_line_break()
print()
print(db.findMax("a").get_denotation())
print(db.findMax("a").get_utterance())
print()
print_line_break()
print()
print(db.findMin("a").get_denotation())
print(db.findMin("a").get_utterance())
print()
print_line_break()


print("                                                         #### --- CALL CORRELATION METHODS --- ####")
print_line_break()
print()
print(db.findCorr("a", "b").get_denotation())
print(db.findCorr("a", "b").get_utterance())
print()
print_line_break()
print()
print(db.largestCorr("a").get_denotation())
print(db.largestCorr("a").get_utterance())
print()
print_line_break()
print()
print(db.largestCorrList("a", 4).get_denotation())
print(db.largestCorrList("a", 4).get_utterance())
print()
print_line_break()
print()
print(db.overallLargestCorrs(6).get_denotation())
print(db.overallLargestCorrs(6).get_utterance())
print()
print_line_break()


print("                                                         #### --- CALL SIMPLE REGRESSION METHODS --- ####")
print_line_break()
print()
print(db.reg("a", "b").get_denotation())
print()
print(db.reg("a", "b").get_utterance())
print()
print_line_break()
print()
print(db.multiReg("a", ["b", "dummy"]).get_denotation())
print()
print(db.multiReg("a", ["b", "dummy"]).get_utterance())
print()
print_line_break()
print()
    # print(db.fixedEffects)
print()

print(db.summarizeLogisticRegression("dummy", np.array(["b", "c"])).get_denotation())
print(db.summarizeLogisticRegression("dummy", np.array(["b", "c"])).get_utterance())
print()
print_line_break()
print()
print(db.logisticMarginalEffects("dummy", np.array(["b", "c"])).get_denotation())
print(db.logisticMarginalEffects("dummy", np.array(["b", "c"])).get_utterance())
print()
print_line_break()


print("                                                         #### --- CALL INSTRUMENTAL VARIABLE METHODS --- ####")
print_line_break()
print()
print(db.ivRegress("b", ["c", "d"], ["a", "dummy"]).get_denotation())
print(db.ivRegress("b", ["c", "d"], ["a", "dummy"]).get_utterance())
print()
print_line_break()
print()
print(db.homoskedasticJStatistic("b", ["c", "d"], ["a", "dummy", "e"]).get_denotation())
print(db.homoskedasticJStatistic("b", ["c", "d"], ["a", "dummy", "e"]).get_utterance())
print()
print_line_break()
print()
print(db.test_weak_instruments("c", ["b", "d", "e"]).get_denotation())
print(db.test_weak_instruments("c", ["b", "d", "e"]).get_utterance())
print()
print_line_break()
print()
print(db.find_instruments("c", np.array(["b", "h"]), "a", np.array(["d", "e", "f", "g", "dummy", "i"])).get_denotation())
print(db.find_instruments("c", np.array(["b", "h"]), "a", np.array(["e", "d", "g", "f", "i", "dummy"])).get_utterance())
print()
print(db.find_instruments("a", np.array(["b", "c"]), "d", np.array(["e", "f", "g", "h", "dummy"])).get_denotation())
print(db.find_instruments("a", np.array(["b", "c"]), "d", np.array(["f", "e", "h", "g", "dummy"])).get_utterance())
print()
print(db.find_instruments("h", np.array(["c", "i"]), "d", np.array(["a", "e", "f", "g"])).get_denotation())
print(db.find_instruments("h", np.array(["c", "i"]), "d", np.array(["e", "f", "g", "a"])).get_utterance())
print()
print(db.find_instruments("e", np.array(["a", "b"]), "h", np.array(["d", "f", "g", "dummy", "i"])).get_denotation())
print(db.find_instruments("e", np.array(["a", "b"]), "h", np.array(["f", "g", "dummy", "i", "d"])).get_utterance())
print()
print_line_break()


print("                                                         #### --- CALL TIME SERIES METHODS --- ####")
print_line_break()
print()
print()

print(db.print_a_bunch_of_AR_shit("a", "year", 4).get_denotation())
print(db.print_a_bunch_of_AR_shit("a", "year", 4).get_utterance())
print()
print_line_break()
print()

    # print(db.AR_with_moving_average("b", 3, 2, "year").get_denotation())
    # print(db.AR_with_moving_average("b", 3, 2, "year").get_utterance())
print()
print_line_break()
print()
print(db.augmented_dicky_fuller_test("c").get_denotation())
print(db.augmented_dicky_fuller_test("c").get_utterance())
print()

print_line_break()
print()
print(db.summarize_VAR(np.array(["a", "c", "dummy"]), "year", 4).get_denotation())
print(db.summarize_VAR(np.array(["a", "c", "dummy"]), "year", 4).get_utterance())
print()
print_line_break()

    # TEST P-VALUE METHOD

print(db.granger_causality_test(["a","c", "dummy"], "year", 3, "a").get_denotation())
print(db.granger_causality_test(["a","c", "dummy"], "year", 3, "a").get_utterance())
print_line_break()

    # try the univariate case:
print(db.analyze_lags("a", "year").get_denotation())
print(db.analyze_lags("a", "year").get_utterance())
print()
print()

    # next try multivariate:
print(db.analyze_lags(["a", "b", "c"], "year", max_lag=11).get_denotation())
print(db.analyze_lags(["a", "b", "c"], "year", max_lag=11).get_utterance())
print()
print_line_break()
print()

#     # try the univariate case:
# print(db.find_optimal_lag_length("a", "year").get_denotation())
# print(db.find_optimal_lag_length("a", "year").get_utterance())
# print()
# print()

#     # next try multivariate:
# print(db.find_optimal_lag_length(["a", "b", "c"], "year", max_lag=11).get_denotation())
# print(db.find_optimal_lag_length(["a", "b", "c"], "year", max_lag=11).get_utterance())
# print()
# print_line_break()
# print()

vars_1 = np.array(["a", "b", "c"])
vars_2 = np.array(["b", "a", "c"])
vars_3 = np.array(["c", "a", "b"])
all_vars = np.array([vars_1, vars_2, vars_3])
print(db.choose_among_regression_models(all_vars).get_denotation())
print(db.choose_among_regression_models(all_vars).get_utterance())
print()
print_line_break()

print("                                                         #### --- CALL ADVANCED STATISTICAL METHODS --- ####")
print_line_break()
print(db.print_PCA_wrapper(["b", "c", "d"]).get_denotation())
print(db.print_PCA_wrapper(["b", "c", "d"]).get_utterance())
print()
print_line_break()
print()
print(db.poisson_regression("f", ["c", "dummy"]).get_denotation())
print(db.poisson_regression("f", ["c", "dummy"]).get_utterance())
print()
print_line_break()
print(db.markov_switching_regime_regression("f", 3).get_denotation())
print(db.markov_switching_regime_regression("f", 3).get_utterance())
print()
print_line_break()
print("                                                         #### --- COMPLETE --- ####")
print_line_break()
print_line_break()
print_line_break()
print_line_break()
print_line_break()
vars_1 = ["a", "c", "dummy"]
analyze_lags = db.analyze_lags(vars_1, "year")
print(analyze_lags.denotation)
print()
'''