import numpy as np
import pandas as pd

from athena_all.sem_parser.grammar import Example
from athena_all.databook import DataBook
from athena_all.file_processing import create_examples_from_amt


class trainingExamples:
    def __init__(self):
        self.train_examples = self.get_train_examples()

    def train_example_matrix(self):
        return pd.DataFrame(
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

    def get_train_examples(self):
        db = DataBook()
        db.add_df(self.train_example_matrix())

        the_five = [
            ["enterq", "dummy"],
            ["tank", "person_id"],
            ["year", "person_id"],
            ["bedwards", "enterq"],
            ["year", "tank"],
        ]
        the_three = [["enterq", "dummy"], ["year", "person_id"], ["bedwards", "enterq"]]
        the_four = [
            ["enterq", "dummy"],
            ["tank", "person_id"],
            ["year", "person_id"],
            ["bedwards", "enterq"],
        ]
        the_six = [
            ["enterq", "dummy"],
            ["tank", "person_id"],
            ["year", "person_id"],
            ["bedwards", "enterq"],
            ["bedwards", "tank"],
            ["year", "tank"],
        ]

        reg_GDP_STINGER = db.reg("gdp", "stinger").denotation
        reg_BEAST_MATIC = db.reg("beast", "matic").denotation
        reg_TANK_BEDWARDS = db.reg("tank", "bedwards").denotation

        dep_vars_1 = ["stinger", "matic", "beast"]
        dep_vars_2 = ["bedwards", "tank", "dummy"]
        dep_vars_3 = ["gdp", "easy", "auto", "matic"]
        dep_vars_4 = ["easy", "matic", "enterq", "year", "animal"]

        multi_reg_1 = db.multiReg("gdp", dep_vars_1).denotation
        multi_reg_2 = db.multiReg("gdp", dep_vars_2).denotation
        multi_reg_3 = db.multiReg("stinger", dep_vars_3).denotation
        multi_reg_4 = db.multiReg("animal", dep_vars_3).denotation

        multi_reg_5 = db.multiReg("bedwards", dep_vars_4).denotation
        multi_reg_6 = db.multiReg("auto", dep_vars_4).denotation

        summarizeLogisticRegression_1 = db.summarizeLogisticRegression(
            "dummy", dep_vars_1
        ).get_denotation()
        # summarizeLogisticRegression_2 = db.summarizeLogisticRegression(
        #     "dummy", dep_vars_2
        # ).get_denotation()
        summarizeLogisticRegression_3 = db.summarizeLogisticRegression(
            "dummy", dep_vars_3
        ).get_denotation()
        summarizeLogisticRegression_4 = db.summarizeLogisticRegression(
            "dummy", dep_vars_4
        ).get_denotation()

        fe_1 = db.fixedEffects(
            "auto", "matic", "person_id", "panel_year"
        ).get_denotation()
        fe_2 = db.fixedEffects(
            "beast", "stinger", "person_id", "panel_year"
        ).get_denotation()
        fe_3 = db.fixedEffects(
            "stinger", "animal", "person_id", "panel_year"
        ).get_denotation()
        fe_4 = db.fixedEffects(
            "bedwards", "beast", "person_id", "panel_year"
        ).get_denotation()

        instruments_1 = ["gdp", "easy", "auto", "dummy"]
        instruments_2 = ["stinger", "matic", "beast", "auto"]
        instruments_3 = ["stinger", "person_id", "dummy", "year", "age"]
        instruments_4 = ["gdp", "panel_year", "dummy", "beast", "tank", "stinger"]

        iv_fit_1 = db.ivRegress("bedwards", dep_vars_1, instruments_1).denotation
        iv_fit_2 = db.ivRegress("gdp", dep_vars_2, instruments_2).denotation
        iv_fit_3 = db.ivRegress("beast", dep_vars_3, instruments_3).denotation
        iv_fit_4 = db.ivRegress("age", dep_vars_4, instruments_4).denotation

        dep_vars_1 = ["stinger", "matic", "beast"]
        dep_vars_2 = ["bedwards", "tank", "dummy"]
        dep_vars_3 = ["gdp", "easy", "auto", "matic"]
        dep_vars_4 = ["easy", "matic", "enterq", "year", "animal"]

        instruments_1 = ["gdp", "easy", "auto", "dummy"]
        instruments_2 = ["stinger", "matic", "beast", "auto"]
        instruments_3 = ["stinger", "person_id", "dummy", "year", "age"]
        instruments_4 = ["gdp", "panel_year", "dummy", "beast", "tank", "stinger"]

        j_stat_1 = db.homoskedasticJStatistic(
            "bedwards", dep_vars_1, instruments_1
        ).denotation
        j_stat_2 = db.homoskedasticJStatistic(
            "gdp", dep_vars_2, instruments_2
        ).denotation
        j_stat_3 = db.homoskedasticJStatistic(
            "beast", dep_vars_3, instruments_3
        ).denotation
        j_stat_4 = db.homoskedasticJStatistic(
            "age", dep_vars_4, instruments_4
        ).denotation

        dicky_fuller_1 = db.augmented_dicky_fuller_test("bedwards").denotation
        dicky_fuller_2 = db.augmented_dicky_fuller_test("stinger").denotation
        dicky_fuller_3 = db.augmented_dicky_fuller_test("gdp").denotation
        dicky_fuller_4 = db.augmented_dicky_fuller_test("tank").denotation

        analyze_lags_1 = db.analyze_lags("matic", "year").denotation
        analyze_lags_2 = db.analyze_lags(dep_vars_2, "year").denotation
        analyze_lags_3 = db.analyze_lags("stinger", "year").denotation

        # find_optimal_lag_length_1 = db.find_optimal_lag_length(
        #     "auto", "year"
        # ).denotation
        # find_optimal_lag_length_2 = db.find_optimal_lag_length(
        #     "matic", "year"
        # ).denotation
        # find_optimal_lag_length_3 = db.find_optimal_lag_length(
        #     ["bedwards", "tank"], "year"
        # ).denotation
        # find_optimal_lag_length_4 = db.find_optimal_lag_length(
        #     ["gdp", "easy"], "year"
        # ).denotation

        poisson_1 = db.poisson_regression("gdp", "stinger").denotation
        poisson_2 = db.poisson_regression("auto", "matic").denotation
        poisson_3 = db.poisson_regression("bedwards", "gdp").denotation
        poisson_4 = db.poisson_regression("tank", "stinger").denotation

        markov_switching_regime_regression_1 = db.markov_switching_regime_regression(
            "stinger", 3
        ).denotation
        markov_switching_regime_regression_2 = db.markov_switching_regime_regression(
            "gdp", 4
        ).denotation
        markov_switching_regime_regression_3 = db.markov_switching_regime_regression(
            "beast", 2
        ).denotation
        markov_switching_regime_regression_4 = db.markov_switching_regime_regression(
            "auto", 5
        ).denotation

        np_list_1 = ["stinger", "matic", "beast"]
        np_list_2 = ["bedwards", "tank", "dummy"]
        np_list_3 = ["gdp", "easy", "auto", "matic"]
        np_list_4 = ["easy", "matic", "enterq", "year", "animal"]

        set_of_vars_1 = np.array([np_list_2, np_list_3, np_list_4])
        set_of_vars_2 = np.array([np_list_1, np_list_3, np_list_4])
        set_of_vars_3 = np.array([np_list_1, np_list_2, np_list_4])
        set_of_vars_4 = np.array([np_list_1, np_list_2, np_list_3])

        choose_among_regression_models_1 = db.choose_among_regression_models(
            set_of_vars_1,
        ).denotation
        choose_among_regression_models_2 = db.choose_among_regression_models(
            set_of_vars_2
        ).denotation
        choose_among_regression_models_3 = db.choose_among_regression_models(
            set_of_vars_3
        ).denotation
        choose_among_regression_models_4 = db.choose_among_regression_models(
            set_of_vars_4
        ).denotation

        test_weak_instruments_1 = db.test_weak_instruments(
            "bedwards", dep_vars_1
        ).denotation
        test_weak_instruments_2 = db.test_weak_instruments(
            "stinger", dep_vars_2
        ).denotation
        test_weak_instruments_3 = db.test_weak_instruments(
            "enterq", dep_vars_3
        ).denotation

        find_instruments_1 = db.find_instruments(
            "tank", "enterq", "bedwards", dep_vars_1
        ).denotation
        find_instruments_2 = db.find_instruments(
            "beast", "enterq", "stinger", dep_vars_2
        ).denotation
        find_instruments_3 = db.find_instruments(
            "bedwards", "enterq", "stinger", dep_vars_3
        ).denotation

        logisticMarginalEffects_1 = db.logisticMarginalEffects(
            "dummy", dep_vars_1
        ).denotation
        # logisticMarginalEffects_2 = db.logisticMarginalEffects("dummy", dep_vars_2).denotation
        logisticMarginalEffects_3 = db.logisticMarginalEffects(
            "dummy", dep_vars_3
        ).denotation
        logisticMarginalEffects_4 = db.logisticMarginalEffects(
            "dummy", dep_vars_4
        ).denotation

        print_a_bunch_of_AR_shit_1 = db.print_a_bunch_of_AR_shit(
            "stinger", "year", 4
        ).denotation
        print_a_bunch_of_AR_shit_2 = db.print_a_bunch_of_AR_shit(
            "beast", "year", 7
        ).denotation
        print_a_bunch_of_AR_shit_3 = db.print_a_bunch_of_AR_shit(
            "auto", "year", 3
        ).denotation
        print_a_bunch_of_AR_shit_4 = db.print_a_bunch_of_AR_shit(
            "matic", "year", 11
        ).denotation

        summarize_VAR_1 = db.summarize_VAR(dep_vars_1, "year", 3).denotation
        summarize_VAR_2 = db.summarize_VAR(dep_vars_2, "year", 2).denotation
        summarize_VAR_3 = db.summarize_VAR(dep_vars_3, "year", 2).denotation

        granger_causality_test_1 = db.granger_causality_test(
            dep_vars_1, "year", 3, "stinger",
        ).denotation
        granger_causality_test_2 = db.granger_causality_test(
            dep_vars_2, "year", 3, "bedwards",
        ).denotation
        granger_causality_test_3 = db.granger_causality_test(
            dep_vars_3, "year", 3, "gdp",
        ).denotation

        print_PCA_wrapper_1 = db.print_PCA_wrapper(["stinger", "beast"]).denotation
        print_PCA_wrapper_2 = db.print_PCA_wrapper(["auto", "matic"]).denotation
        print_PCA_wrapper_3 = db.print_PCA_wrapper(["dummy", "enterq"]).denotation

        denoted_examples = [
            # --------------------------------------------------------------------------------------------------------------- #
            # for AMT:
            Example(
                input="How are you?",
                semantics=("greeting",),
                to_lower=True,
                denotation=69,
            ),
            Example(
                input="Can you help me find a correlation?",
                semantics=("help", "findCorr"),
                to_lower=True,
                denotation=6969,
            ),
            Example(
                input="What seems interesting?",
                semantics=("summarize",),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Can you find some structure in the data?",
                semantics=("summarize",),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Tell me about the dataset.",
                semantics=("summarize",),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Can you summarize the data?",
                semantics=("summarize",),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Are you sure that's right?",
                semantics=("check_answer",),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Could you try that again?",
                semantics=("check_answer",),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="That's not what I wanted.",
                semantics=("incorrect_answer",),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="I don't think that's right",
                semantics=("incorrect_answer",),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Perfect that looks great. What else can you do?",
                semantics=("thanks", "help", "functionality"),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Thanks so much!",
                semantics=("thanks"),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Do you know how to find p-statistics?",
                semantics=("help", "findPstat"),
                to_lower=True,
                denotation=6969,
            ),
            Example(
                input="Can I upload a different dataset?",
                semantics=("different_dataset",),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Can you explain what regression does?",
                semantics=("help", "reg"),
                to_lower=True,
                denotation=6969,
            ),
            Example(
                input="Help, how does this all work.",
                semantics=("help"),
                to_lower=True,
                denotation=6969,
            ),
            Example(
                input="How did you do that?",
                semantics=("help", "prev_result"),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="What did you just do there?",
                semantics=("help", "prev_result"),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="How do I use the regression feature?",
                semantics=("help", "reg"),
                to_lower=True,
                denotation=6969,
            ),
            Example(
                input="Graph stinger versus auto for me please.",
                semantics=("graph", "args"),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="run an OLS regression of stinger on bedwards",
                semantics=("reg", "stinger", "bedwards"),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Which five cols are most correlated with stinger",
                semantics=("largestCorrList", "stinger", 5),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Which of the columns has the largest correlation with stinger",
                semantics=("largestCorr", "BEDWARDS"),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Show the three biggest values of bedwards.",
                semantics=("summarize",),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="Calculate the variance of bedwards, please.",
                semantics=("findVar", "bedwards"),
                to_lower=True,
                denotation=0,
            ),
            Example(
                input="regress auto on EASY, MATIC, ENTERQ, YEAR, ANIMAL",
                semantics=("multiReg", "auto", dep_vars_4),
                to_lower=True,
                denotation=multi_reg_5,
            ),
            Example(
                input="Run linear regression of stinger on GDP, EASY, AUTO, and MATIC.",
                semantics=("multiReg", "stinger", dep_vars_3),
                to_lower=True,
                denotation=multi_reg_4,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # findMean: proper grammar
            Example(
                input="Find the mean of ENTERQ",
                semantics=("findMean", "enterq"),
                to_lower=True,
                denotation=0.4654046005555556,
            ),
            Example(
                input="What is the mean of AGE",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            Example(
                input="Find the average of ENTERQ",
                semantics=("findMean", "enterq"),
                to_lower=True,
                denotation=0.4654046005555556,
            ),
            Example(
                input="What is the average of AGE",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            Example(
                input="Find the avg of ENTERQ",
                semantics=("findMean", "enterq"),
                to_lower=True,
                denotation=0.4654046005555556,
            ),
            Example(
                input="What is the avg of AGE",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            # findMean: improper grammar
            Example(
                input="Show mean of ENTERQ",
                semantics=("findMean", "enterq"),
                to_lower=True,
                denotation=0.4654046005555556,
            ),
            Example(
                input="mean of AGE",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            Example(
                input="Find the average of ENTERQ",
                semantics=("findMean", "enterq"),
                to_lower=True,
                denotation=0.4654046005555556,
            ),
            Example(
                input="Whats the average of AGE",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            Example(
                input="What's the mean of AGE",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            Example(
                input="average of ENTERQ",
                semantics=("findMean", "enterq"),
                to_lower=True,
                denotation=0.4654046005555556,
            ),
            Example(
                input="average AGE",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            Example(
                input="AGE's mean",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            Example(
                input="AGEs mean",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            Example(
                input="AGE's average",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            Example(
                input="Show AGE's mean",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            Example(
                input="Find AGEs mean",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            Example(
                input="What is AGE's average",
                semantics=("findMean", "age"),
                to_lower=True,
                denotation=0.5918052762222222,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # findStd: proper gramamr
            Example(
                input="Find the std of ENTERQ",
                semantics=("findStd", "ENTERQ"),
                to_lower=True,
                denotation=0.317088313970524,
            ),
            Example(
                input="What is the standard deviation of AGE",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            Example(
                input="Find the standard deviation of ENTERQ",
                semantics=("findStd", "enterq"),
                to_lower=True,
                denotation=0.317088313970524,
            ),
            Example(
                input="What is the std of AGE",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            Example(
                input="Find the std of ENTERQ",
                semantics=("findStd", "enterq"),
                to_lower=True,
                denotation=0.317088313970524,
            ),
            Example(
                input="What is the standard dev of AGE",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            # findStd: improper grammar
            Example(
                input="Show standard deviation of ENTERQ",
                semantics=("findStd", "enterq"),
                to_lower=True,
                denotation=0.317088313970524,
            ),
            Example(
                input="std of AGE",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            Example(
                input="Find the std of ENTERQ",
                semantics=("findStd", "enterq"),
                to_lower=True,
                denotation=0.317088313970524,
            ),
            Example(
                input="Whats the std deviation of AGE",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            Example(
                input="What's the stddev of AGE",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            Example(
                input="standard deviation of ENTERQ",
                semantics=("findStd", "enterq"),
                to_lower=True,
                denotation=0.317088313970524,
            ),
            Example(
                input="standard deviation AGE",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            Example(
                input="AGE's stddev",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            Example(
                input="AGEs stddev",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            Example(
                input="AGE's standard deviation",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            Example(
                input="Show AGE's stddev",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            Example(
                input="Find AGEs stddev",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            Example(
                input="What is AGE's standard deviation",
                semantics=("findStd", "age"),
                to_lower=True,
                denotation=0.24739176853948733,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # findVar: proper grammar
            Example(
                input="Find the var of ENTERQ",
                semantics=("findVar", "enterq"),
                to_lower=True,
                denotation=0.10054499885666962,
            ),
            Example(
                input="What is the variance of AGE",
                semantics=("findVar", "age"),
                to_lower=True,
                denotation=0.06120268714109527,
            ),
            Example(
                input="Find the spread of ENTERQ",
                semantics=("findVar", "enterq"),
                to_lower=True,
                denotation=0.10054499885666962,
            ),
            Example(
                input="What is the var of AGE",
                semantics=("findVar", "age"),
                to_lower=True,
                denotation=0.06120268714109527,
            ),
            Example(
                input="Find the variance of ENTERQ",
                semantics=("findVar", "enterq"),
                to_lower=True,
                denotation=0.10054499885666962,
            ),
            Example(
                input="What is the var of AGE",
                semantics=("findVar", "age"),
                to_lower=True,
                denotation=0.06120268714109527,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # findMax: proper grammar
            Example(
                input="Find the max of ENTERQ",
                semantics=("findMax", "enterq"),
                to_lower=True,
                denotation=0.95031834,
            ),
            Example(
                input="What is the max of AGE",
                semantics=("findMax", "age"),
                to_lower=True,
                denotation=0.9249519,
            ),
            Example(
                input="Find the max value in ENTERQ",
                semantics=("findMax", "enterq"),
                to_lower=True,
                denotation=0.95031834,
            ),
            Example(
                input="What is the biggest value in AGE",
                semantics=("findMax", "age"),
                to_lower=True,
                denotation=0.9249519,
            ),
            Example(
                input="Find the largest ENTERQ value",
                semantics=("findMax", "enterq"),
                to_lower=True,
                denotation=0.95031834,
            ),
            Example(
                input="What is the biggest entry in AGE",
                semantics=("findMax", "AGE"),
                to_lower=True,
                denotation=0.9249519,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # findMin: proper grammar
            Example(
                input="Find the min of ENTERQ",
                semantics=("findMin", "enterq"),
                to_lower=True,
                denotation=0.0038075,
            ),
            Example(
                input="What is the min of AGE",
                semantics=("findMin", "age"),
                to_lower=True,
                denotation=0.0548202,
            ),
            Example(
                input="Find the min value in ENTERQ",
                semantics=("findMin", "enterq"),
                to_lower=True,
                denotation=0.0038075,
            ),
            Example(
                input="What is the smallest value in AGE",
                semantics=("findMin", "age"),
                to_lower=True,
                denotation=0.0548202,
            ),
            Example(
                input="Find the smallest ENTERQ value",
                semantics=("findMin", "enterq"),
                to_lower=True,
                denotation=0.0038075,
            ),
            Example(
                input="What is the smallest entry in AGE",
                semantics=("findMin", "age"),
                to_lower=True,
                denotation=0.0548202,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # findMedian: proper grammar
            Example(
                input="Find the median of ENTERQ",
                semantics=("findMedian", "enterq"),
                to_lower=True,
                denotation=0.419915855,
            ),
            Example(
                input="What is the median of AGE",
                semantics=("findMedian", "age"),
                to_lower=True,
                denotation=0.6287337550000001,
            ),
            Example(
                input="Find the median value in ENTERQ",
                semantics=("findMedian", "enterq"),
                to_lower=True,
                denotation=0.419915855,
            ),
            Example(
                input="What is the middle value in AGE",
                semantics=("findMedian", "age"),
                to_lower=True,
                denotation=0.6287337550000001,
            ),
            Example(
                input="Find the middle ENTERQ value",
                semantics=("findMedian", "enterq"),
                to_lower=True,
                denotation=0.419915855,
            ),
            Example(
                input="What is the median entry in AGE",
                semantics=("findMedian", "age"),
                to_lower=True,
                denotation=0.6287337550000001,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # findCorr: proper grammar
            Example(
                input="Find the correlation of ENTERQ and AGE",
                semantics=("findCorr", "enterq", "age"),
                to_lower=True,
                denotation=0.34482057605950656,
            ),
            Example(
                input="Find the correlation of AGE and ENTERQ",
                semantics=("findCorr", "enterq", "age"),
                to_lower=True,
                denotation=0.34482057605950656,
            ),
            Example(
                input="What's the correlation between tank and BEAST",
                semantics=("findCorr", "tank", "beast"),
                to_lower=True,
                denotation=0.15523942673478727,
            ),
            Example(
                input="Find corr between BEAST and TANK",
                semantics=("findCorr", "tank", "beast"),
                to_lower=True,
                denotation=0.15523942673478727,
            ),
            Example(
                input="What's the corr between AUTO and MATIC",
                semantics=("findCorr", "auto", "matic"),
                to_lower=True,
                denotation=0.2713160561112919,
            ),
            Example(
                input="What's the correlation across MATIC and AUTO",
                semantics=("findCorr", "auto", "matic"),
                to_lower=True,
                denotation=0.2713160561112919,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # largestCorr: proper grammar
            Example(
                input="Which variable is most correlated with ENTERQ",
                semantics=("largestCorr", "enterq"),
                to_lower=True,
                denotation="DUMMY",
            ),
            Example(
                input="Which column is most correlated with ENTERQ",
                semantics=("largestCorr", "enterq"),
                to_lower=True,
                denotation="DUMMY",
            ),
            Example(
                input="What column has the largest corr with TANK",
                semantics=("largestCorr", "tank"),
                to_lower=True,
                denotation="person_id",
            ),
            Example(
                input="What's most correlated with TANK",
                semantics=("largestCorr", "tank"),
                to_lower=True,
                denotation="person_id",
            ),
            Example(
                input="What var is the most correlated with BEDWARDS",
                semantics=("largestCorr", "bedwards"),
                to_lower=True,
                denotation="ENTERQ",
            ),
            Example(
                input="Which of the columns has the largest correlation with BEDWARDS",
                semantics=("largestCorr", "bedwards"),
                to_lower=True,
                denotation="ENTERQ",
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # largestCorrList: proper grammar
            Example(
                input="Which three variables are most correlated with ENTERQ",
                semantics=("largestCorrList", "enterq", 3),
                to_lower=True,
                denotation=["DUMMY", "TANK", "BEDWARDS"],
            ),
            Example(
                input="What four columns are most correlated with ENTERQ",
                semantics=("largestCorrList", "enterq", 4),
                to_lower=True,
                denotation=["DUMMY", "TANK", "AGE", "BEDWARDS"],
            ),
            Example(
                input="Which 5 columns have the largest corr with ENTERQ",
                semantics=("largestCorrList", "enterq", 5),
                to_lower=True,
                denotation=["DUMMY", "TANK", "BEAST", "AGE", "BEDWARDS"],
            ),
            Example(
                input="Which 3 variables are most correlated with MATIC",
                semantics=("largestCorrList", "matic", 3),
                to_lower=True,
                denotation=["GDP", "STINGER", "BEDWARDS"],
            ),
            Example(
                input="What four vars have the largest corr with MATIC",
                semantics=("largestCorrList", "matic", 4),
                to_lower=True,
                denotation=["GDP", "STINGER", "BEDWARDS", "AUTO"],
            ),
            Example(
                input="Which five cols are most correlated with MATIC",
                semantics=("largestCorrList", "matic", 5),
                to_lower=True,
                denotation=["GDP", "STINGER", "BEDWARDS", "AUTO", "PANEL_YEAR"],
            ),
            Example(
                input="Which 3 columns are most correlated with TANK",
                semantics=("largestCorrList", "tank", 3),
                to_lower=True,
                denotation=["PERSON_ID", "BEDWARDS", "YEAR"],
            ),
            Example(
                input="What 4 variables have the largest corr with TANK",
                semantics=("largestCorrList", "tank", 4),
                to_lower=True,
                denotation=["PERSON_ID", "ENTERQ", "BEDWARDS", "YEAR"],
            ),
            Example(
                input="Which 5 variables are the most correlated with TANK",
                semantics=("largestCorrList", "tank", 5),
                to_lower=True,
                denotation=["PERSON_ID", "ENTERQ", "BEDWARDS", "AUTO", "YEAR"],
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # overallLargestCorrs: proper grammar
            Example(
                input="What are the largest correlations in the dataset",
                semantics=("overallLargestCorrs"),
                to_lower=True,
                denotation=the_five,
            ),
            Example(
                input="What are some of the most important relationships",
                semantics=("overallLargestCorrs"),
                to_lower=True,
                denotation=the_five,
            ),
            Example(
                input="What are the biggest corrs in the dataset",
                semantics=("overallLargestCorrs"),
                to_lower=True,
                denotation=the_five,
            ),
            Example(
                input="What are the three largest correlations in the dataset",
                semantics=("overallLargestCorrs", 3),
                to_lower=True,
                denotation=the_three,
            ),
            Example(
                input="What are the 3 most important relationships",
                semantics=("overallLargestCorrs", 3),
                to_lower=True,
                denotation=the_three,
            ),
            Example(
                input="What are the 3 biggest corrs in the dataset",
                semantics=("overallLargestCorrs", 3),
                to_lower=True,
                denotation=the_three,
            ),
            Example(
                input="What are the 4 largest correlations in the dataset",
                semantics=("overallLargestCorrs", 4),
                to_lower=True,
                denotation=the_four,
            ),
            Example(
                input="What are the four most important relationships",
                semantics=("overallLargestCorrs", 4),
                to_lower=True,
                denotation=the_four,
            ),
            Example(
                input="What are the 4 biggest corrs in the dataset",
                semantics=("overallLargestCorrs", 4),
                to_lower=True,
                denotation=the_four,
            ),
            Example(
                input="What are the 6 largest correlations in the dataset",
                semantics=("overallLargestCorrs", 6),
                to_lower=True,
                denotation=the_six,
            ),
            Example(
                input="What are the six most important relationships?",
                semantics=("overallLargestCorrs", 6),
                to_lower=True,
                denotation=the_six,
            ),
            Example(
                input="What are the six biggest corrs in the dataset",
                semantics=("overallLargestCorrs", 6),
                to_lower=True,
                denotation=the_six,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # reg: proper grammar
            Example(
                input="regress GDP on STINGER",
                semantics=("reg", "gdp", "stinger"),
                to_lower=True,
                denotation=reg_GDP_STINGER,
            ),
            Example(
                input="reg GDP on STINGER",
                semantics=("reg", "gdp", "stinger"),
                to_lower=True,
                denotation=reg_GDP_STINGER,
            ),
            Example(
                input="run a regression of GDP on STINGER",
                semantics=("reg", "gdp", "stinger"),
                to_lower=True,
                denotation=reg_GDP_STINGER,
            ),
            Example(
                input="regress BEAST on MATIC",
                semantics=("reg", "beast", "matic"),
                to_lower=True,
                denotation=reg_BEAST_MATIC,
            ),
            Example(
                input="reg BEAST on MATIC",
                semantics=("reg", "beast", "matic"),
                to_lower=True,
                denotation=reg_BEAST_MATIC,
            ),
            Example(
                input="linear regression of BEAST on MATIC",
                semantics=("reg", "beast", "matic"),
                to_lower=True,
                denotation=reg_BEAST_MATIC,
            ),
            Example(
                input="regress TANK on BEDWARDS",
                semantics=("reg", "tank", "bedwards"),
                to_lower=True,
                denotation=reg_TANK_BEDWARDS,
            ),
            Example(
                input="reg TANK on BEDWARDS",
                semantics=("reg", "tank", "bedwards"),
                to_lower=True,
                denotation=reg_TANK_BEDWARDS,
            ),
            Example(
                input="run OLS regression of TANK on MATIC",
                semantics=("reg", "tank", "bedwards"),
                to_lower=True,
                denotation=reg_TANK_BEDWARDS,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # multiReg: proper grammar
            Example(
                input="regress GDP on STINGER, MATIC, and BEAST",
                semantics=("multiReg", "gdp", dep_vars_1),
                to_lower=True,
                denotation=multi_reg_1,
            ),
            Example(
                input="regress GDP on BEDWARDS, TANK, and DUMMY",
                semantics=("multiReg", "gdp", dep_vars_2),
                to_lower=True,
                denotation=multi_reg_2,
            ),
            Example(
                input="regress STINGER on GDP, EASY, AUTO, and MATIC",
                semantics=("multiReg", "stinger", dep_vars_3),
                to_lower=True,
                denotation=multi_reg_3,
            ),
            Example(
                input="regress ANIMAL on GDP, EASY, AUTO, and MATIC",
                semantics=("multiReg", "animal", dep_vars_3),
                to_lower=True,
                denotation=multi_reg_4,
            ),
            Example(
                input="regress BEDWARDS on EASY, MATIC, ENTERQ, YEAR, ANIMAL",
                semantics=("multiReg", "bedwards", dep_vars_4),
                to_lower=True,
                denotation=multi_reg_5,
            ),
            Example(
                input="regress AUTO on EASY, MATIC, ENTERQ, YEAR, ANIMAL",
                semantics=("multiReg", "auto", dep_vars_4),
                to_lower=True,
                denotation=multi_reg_6,
            ),
            Example(
                input="reg GDP on STINGER, MATIC, and BEAST",
                semantics=("multiReg", "gdp", dep_vars_1),
                to_lower=True,
                denotation=multi_reg_1,
            ),
            Example(
                input="run a regression of GDP on BEDWARDS, TANK, and DUMMY",
                semantics=("multiReg", "gdp", dep_vars_2),
                to_lower=True,
                denotation=multi_reg_2,
            ),
            Example(
                input="run multivariate regression STINGER on GDP, EASY, AUTO, and MATIC",
                semantics=("multiReg", "stinger", dep_vars_3),
                to_lower=True,
                denotation=multi_reg_3,
            ),
            Example(
                input="run linear regression of ANIMAL on GDP, EASY, AUTO, and MATIC",
                semantics=("multiReg", "animal", dep_vars_3),
                to_lower=True,
                denotation=multi_reg_4,
            ),
            Example(
                input="reg BEDWARDS on EASY, MATIC, ENTERQ, YEAR, ANIMAL",
                semantics=("multiReg", "bedwards", dep_vars_4),
                to_lower=True,
                denotation=multi_reg_5,
            ),
            Example(
                input="multivariable reg AUTO on EASY, MATIC, ENTERQ, YEAR, ANIMAL",
                semantics=("multiReg", "auto", dep_vars_4),
                to_lower=True,
                denotation=multi_reg_6,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # fixedEffects: proper grammar
            # fe_1 = db.fixedEffects("auto", "matic", "person_id", "panel_year").get_denotation()
            # fe_2 = db.fixedEffects("beast", "stinger", "person_id", "panel_year").get_denotation()
            # fe_3 = db.fixedEffects("stinger", "animal", "person_id", "panel_year").get_denotation()
            # fe_4 = db.fixedEffects("bedwards", "beast", "person_id", "panel_year").get_denotation()
            Example(
                input="run a fixed effects regression of auto on matic with person_id as the id and panel_year as the year",
                semantics=("fixedEffects", "auto", "matic", "person_id", "panel_year"),
                to_lower=True,
                denotation=fe_1,
            ),
            Example(
                input="run a longitudinal regression of beast on stinger with person_id as the id dimension and panel_year as the year",
                semantics=(
                    "fixedEffects",
                    "beast",
                    "stinger",
                    "person_id",
                    "panel_year",
                ),
                to_lower=True,
                denotation=fe_2,
            ),
            Example(
                input="run a panel regression of stinger on anima with person_id as the id and panel_year as the year axis",
                semantics=(
                    "fixedEffects",
                    "stinger",
                    "animal",
                    "person_id",
                    "panel_year",
                ),
                to_lower=True,
                denotation=fe_3,
            ),
            Example(
                input="run a fixed effects regression of bedwards on beast with person_id as the id and panel_year as the year dimension",
                semantics=(
                    "fixedEffects",
                    "bedwards",
                    "beast",
                    "person_id",
                    "panel_year",
                ),
                to_lower=True,
                denotation=fe_4,
            ),
            Example(
                input="can you use a fixed effects regression with auto as dependent, matic as independent, and with person_id as the id and panel_year as the year",
                semantics=("fixedEffects", "auto", "matic", "person_id", "panel_year"),
                to_lower=True,
                denotation=fe_1,
            ),
            Example(
                input="run a panel regression of beast on stinger with person_id as the id dimension and panel_year as the year",
                semantics=(
                    "fixedEffects",
                    "beast",
                    "stinger",
                    "person_id",
                    "panel_year",
                ),
                to_lower=True,
                denotation=fe_2,
            ),
            Example(
                input="longitudinal regression of stinger on anima with person_id as the id and panel_year as the year axis",
                semantics=(
                    "fixedEffects",
                    "stinger",
                    "animal",
                    "person_id",
                    "panel_year",
                ),
                to_lower=True,
                denotation=fe_3,
            ),
            Example(
                input="fixed effects regression of bedwards on beast with person_id as the id and panel_year as the year dimension",
                semantics=(
                    "fixedEffects",
                    "bedwards",
                    "beast",
                    "person_id",
                    "panel_year",
                ),
                to_lower=True,
                denotation=fe_4,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # summarizeLogisticRegression: proper grammar
            # summarizeLogisticRegression_1 = db.summarizeLogisticRegression("dummy", dep_vars_1).get_denotation()
            # summarizeLogisticRegression_2 = db.summarizeLogisticRegression("dummy", dep_vars_2).get_denotation()
            # summarizeLogisticRegression_3 = db.summarizeLogisticRegression("dummy", dep_vars_3).get_denotation()
            # summarizeLogisticRegression_4 = db.summarizeLogisticRegression("dummy", dep_vars_4).get_denotation()
            Example(
                input="run a logistic regression of DUMMY on STINGER, MATIC, and BEAST",
                semantics=("summarizeLogisticRegression", "dummy", dep_vars_1),
                to_lower=True,
                denotation=summarizeLogisticRegression_1,
            ),
            # Example(
            #     input="run a logistic regression of DUMMY on BEDWARDS, TANK, and MATIC",
            #     semantics=("summarizeLogisticRegression", "dummy", dep_vars_2),
            #     to_lower=True,
            #     denotation=summarizeLogisticRegression_2,
            # ),
            Example(
                input="run a logistic regression of DUMMY on GDP, EASY, AUTO, and MATIC",
                semantics=("summarizeLogisticRegression", "dummy", dep_vars_3),
                to_lower=True,
                denotation=summarizeLogisticRegression_3,
            ),
            Example(
                input="run a logistic regression of DUMMY on EASY, MATIC, ENTERQ, YEAR, and ANIMAL",
                semantics=("summarizeLogisticRegression", "dummy", dep_vars_4),
                to_lower=True,
                denotation=summarizeLogisticRegression_4,
            ),
            Example(
                input="logistic regression of DUMMY on STINGER, MATIC, and BEAST",
                semantics=("summarizeLogisticRegression", "dummy", dep_vars_1),
                to_lower=True,
                denotation=summarizeLogisticRegression_1,
            ),
            # Example(
            #     input="run binary reg of DUMMY on BEDWARDS, TANK, and MATIC",
            #     semantics=("summarizeLogisticRegression", "dummy", dep_vars_2),
            #     to_lower=True,
            #     denotation=summarizeLogisticRegression_2,
            # ),
            # Example(
            #     input="run a binary regression of DUMMY on BEDWARDS, TANK, and MATIC",
            #     semantics=("summarizeLogisticRegression", "dummy", dep_vars_2),
            #     to_lower=True,
            #     denotation=summarizeLogisticRegression_2,
            # ),
            Example(
                input="run a classification regression of DUMMY on GDP, EASY, AUTO, and MATIC",
                semantics=("summarizeLogisticRegression", "dummy", dep_vars_3),
                to_lower=True,
                denotation=summarizeLogisticRegression_3,
            ),
            Example(
                input="logistic reg DUMMY on EASY, MATIC, ENTERQ, YEAR, and ANIMAL",
                semantics=("summarizeLogisticRegression", "dummy", dep_vars_4),
                to_lower=True,
                denotation=summarizeLogisticRegression_4,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # logisticMarginalEffects: proper grammar
            # logisticMarginalEffects_1 = db.logisticMarginalEffects("dummy", dep_vars_1).denotation
            # UNUSED: logisticMarginalEffects_2 = db.logisticMarginalEffects("dummy", dep_vars_2).denotation
            # logisticMarginalEffects_3 = db.logisticMarginalEffects("dummy", dep_vars_3).denotation
            # logisticMarginalEffects_4 = db.logisticMarginalEffects("dummy", dep_vars_4).denotation
            Example(
                input="run a logistic regression of DUMMY on STINGER, MATIC, and BEAST and show the marginal effects",
                semantics=("logisticMarginalEffects", "dummy", dep_vars_1),
                to_lower=True,
                denotation=logisticMarginalEffects_1,
            ),
            Example(
                input="What are the marginal effects in a logistic regression of DUMMY on GDP, EASY, AUTO, and MATIC",
                semantics=("logisticMarginalEffects", "dummy", dep_vars_3),
                to_lower=True,
                denotation=logisticMarginalEffects_3,
            ),
            Example(
                input="Find the marginal effects of a logistic regression of DUMMY on EASY, MATIC, ENTERQ, YEAR, and ANIMAL",
                semantics=("logisticMarginalEffects", "dummy", dep_vars_4),
                to_lower=True,
                denotation=logisticMarginalEffects_4,
            ),
            Example(
                input="Find the marginal effects of a logistic regression of DUMMY on STINGER, MATIC, and BEAST",
                semantics=("logisticMarginalEffects", "dummy", dep_vars_1),
                to_lower=True,
                denotation=logisticMarginalEffects_1,
            ),
            Example(
                input="Find the marginal effects in a binary regression of DUMMY on GDP, EASY, AUTO, and MATIC",
                semantics=("logisticMarginalEffects", "dummy", dep_vars_3),
                to_lower=True,
                denotation=logisticMarginalEffects_3,
            ),
            Example(
                input="What are the marginal effects of a classification regression of DUMMY on EASY, MATIC, ENTERQ, YEAR, and ANIMAL",
                semantics=("logisticMarginalEffects", "dummy", dep_vars_4),
                to_lower=True,
                denotation=logisticMarginalEffects_4,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # ivRegress: proper grammar
            Example(
                input="regress BEDWARDS on STINGER, MATIC, and BEAST with GDP, EASY, AUTO, and DUMMY as instrumental variables",
                semantics=("ivRegress", "bedwards", dep_vars_1, instruments_1),
                to_lower=True,
                denotation=iv_fit_1,
            ),
            Example(
                input="regress GDP on BEDWARDS, TANK, and DUMMY with STINGER, MATIC, BEAST, and AUTO as instrumental variables",
                semantics=("ivRegress", "GDP", dep_vars_2, instruments_2),
                to_lower=True,
                denotation=iv_fit_2,
            ),
            Example(
                input="regress BEAST on GDP, EASY, AUTO, and MATIC with STINGER, PERSON_ID, DUMMY, YEAR, and AGE as instrumental variables",
                semantics=("ivRegress", "beast", dep_vars_3, instruments_3),
                to_lower=True,
                denotation=iv_fit_3,
            ),
            Example(
                input="regress AGE on EASY, MATIC, ENTERQ, YEAR, and ANIMAL with GDP, PANEL_YEAR, DUMMY, BEAST, TANK and STINGER as instrumental variables",
                semantics=("ivRegress", "age", dep_vars_4, instruments_4),
                to_lower=True,
                denotation=iv_fit_4,
            ),
            Example(
                input="run an instrumental variable regression of BEDWARDS on STINGER, MATIC, and BEAST with GDP, EASY, AUTO, and DUMMY as instruments",
                semantics=("ivRegress", "bedwards", dep_vars_1, instruments_1),
                to_lower=True,
                denotation=iv_fit_1,
            ),
            Example(
                input="reg GDP on BEDWARDS, TANK, and DUMMY with STINGER, MATIC, BEAST, and AUTO as instruments",
                semantics=("ivRegress", "GDP", dep_vars_2, instruments_2),
                to_lower=True,
                denotation=iv_fit_2,
            ),
            Example(
                input="iv regression of BEAST on GDP, EASY, AUTO, and MATIC with STINGER, PERSON_ID, DUMMY, YEAR, and AGE as instrumental variables",
                semantics=("ivRegress", "beast", dep_vars_3, instruments_3),
                to_lower=True,
                denotation=iv_fit_3,
            ),
            Example(
                input="linear instrumental regression of AGE on EASY, MATIC, ENTERQ, YEAR, and ANIMAL with GDP, PANEL_YEAR, DUMMY, BEAST, TANK and STINGER as instrumentals",
                semantics=("ivRegress", "age", dep_vars_4, instruments_4),
                to_lower=True,
                denotation=iv_fit_4,
            ),
            Example(
                input="Use GDP, EASY, AUTO, and DUMMY as instrumental variables to reg BEDWARDS on STINGER, MATIC, and BEAST",
                semantics=("ivRegress", "bedwards", dep_vars_1, instruments_1),
                to_lower=True,
                denotation=iv_fit_1,
            ),
            Example(
                input="Use STINGER, MATIC, BEAST, and AUTO as instrumetns to reg of GDP on BEDWARDS, TANK, and DUMMY",
                semantics=("ivRegress", "GDP", dep_vars_2, instruments_2),
                to_lower=True,
                denotation=iv_fit_2,
            ),
            Example(
                input="With STINGER, PERSON_ID, DUMMY, YEAR, and AGE as instrumental variables regress BEAST on GDP, EASY, AUTO, and MATIC",
                semantics=("ivRegress", "beast", dep_vars_3, instruments_3),
                to_lower=True,
                denotation=iv_fit_3,
            ),
            Example(
                input="Using GDP, PANEL_YEAR, DUMMY, BEAST, TANK and STINGER as instrumental variables regress AGE on EASY, MATIC, ENTERQ, YEAR, and ANIMAL",
                semantics=("ivRegress", "age", dep_vars_4, instruments_4),
                to_lower=True,
                denotation=iv_fit_4,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # homoskedasticJStatistic: proper grammar
            # j_stat_1 = db.homoskedasticJStatistic("bedwards", dep_vars_1, instruments_1).denotation
            # j_stat_2 = db.homoskedasticJStatistic("gdp", dep_vars_2, instruments_2).denotation
            # j_stat_3 = db.homoskedasticJStatistic("beast", dep_vars_3, instruments_3).denotation
            # j_stat_4 = db.homoskedasticJStatistic("age", dep_vars_4, instruments_4).denotation
            # dep_vars_1 = ["stinger", "matic", "beast"]
            # dep_vars_2 = ["bedwards", "tank", "matic"]
            # dep_vars_3 = ["gdp", "easy", "auto", "matic"]
            # dep_vars_4 = ["easy", "matic", "enterq", "year", "animal"]
            # instruments_1 = ["gdp", "easy", "auto", "dummy"]
            # instruments_2 = ["stinger", "matic", "beast", "auto"]
            # instruments_3 = ["stinger", "person_id", "dummy", "year", "age"]
            # instruments_4 = ["gdp", "panel_year", "dummy", "beast", "tank", "stinger"]
            Example(
                input="Find Sargan-Hansen test of BEDWARDS on STINGER, MATIC, and BEAST with GDP, EASY, AUTO, and DUMMY as instrumental variables",
                semantics=(
                    "homoskedasticJStatistic",
                    "bedwards",
                    dep_vars_1,
                    instruments_1,
                ),
                to_lower=True,
                denotation=j_stat_1,
            ),
            Example(
                input="j statistic of regression GDP on BEDWARDS, TANK, and DUMMY with STINGER, MATIC, BEAST, and AUTO as instrumental variables",
                semantics=("homoskedasticJStatistic", "gdp", dep_vars_2, instruments_2),
                to_lower=True,
                denotation=j_stat_2,
            ),
            Example(
                input="test the exogeneity of a regression of BEAST on GDP, EASY, AUTO, and MATIC with STINGER, PERSON_ID, DUMMY, YEAR, and AGE as instrumental variables",
                semantics=(
                    "homoskedasticJStatistic",
                    "beast",
                    dep_vars_3,
                    instruments_3,
                ),
                to_lower=True,
                denotation=j_stat_3,
            ),
            Example(
                input="are the instruments valid in a reg of AGE on EASY, MATIC, ENTERQ, YEAR, and ANIMAL with GDP, PANEL_YEAR, DUMMY, BEAST, TANK and STINGER as instrumental variables",
                semantics=("homoskedasticJStatistic", "age", dep_vars_4, instruments_4),
                to_lower=True,
                denotation=j_stat_4,
            ),
            Example(
                input="j statistic of an instrumental variable regression of BEDWARDS on STINGER, MATIC, and BEAST with GDP, EASY, AUTO, and DUMMY as instruments",
                semantics=(
                    "homoskedasticJStatistic",
                    "bedwards",
                    dep_vars_1,
                    instruments_1,
                ),
                to_lower=True,
                denotation=j_stat_1,
            ),
            Example(
                input="find the j-stat of a regression GDP on BEDWARDS, TANK, and DUMMY with STINGER, MATIC, BEAST, and AUTO as instruments",
                semantics=("homoskedasticJStatistic", "gdp", dep_vars_2, instruments_2),
                to_lower=True,
                denotation=j_stat_2,
            ),
            Example(
                input="what's the Sargan-Hansen statistic of a regression of BEAST on GDP, EASY, AUTO, and MATIC with STINGER, PERSON_ID, DUMMY, YEAR, and AGE as instrumental variables",
                semantics=(
                    "homoskedasticJStatistic",
                    "beast",
                    dep_vars_3,
                    instruments_3,
                ),
                to_lower=True,
                denotation=j_stat_3,
            ),
            Example(
                input="Find the j-statistic instrumental in regression of AGE on EASY, MATIC, ENTERQ, YEAR, and ANIMAL with GDP, PANEL_YEAR, DUMMY, BEAST, TANK and STINGER as instrumentals",
                semantics=("homoskedasticJStatistic", "age", dep_vars_4, instruments_4),
                to_lower=True,
                denotation=j_stat_4,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # test_weak_instruments: proper grammar
            # test_weak_instruments_1 = db.test_weak_instruments("bedwards", dep_vars_1).denotation
            # test_weak_instruments_2 = db.test_weak_instruments("stinger", dep_vars_2).denotation
            # test_weak_instruments_3 = db.test_weak_instruments("enterq", dep_vars_3).denotation
            # dep_vars_1 = ["stinger", "matic", "beast"]
            # dep_vars_2 = ["bedwards", "tank", "dummy"]
            # dep_vars_3 = ["gdp", "easy", "auto", "matic"]
            Example(
                input="Test the strength of instrument candidates stinger, matic, and beast in a regression on bedwards",
                semantics=("test_weak_instruments", "bedwards", dep_vars_1),
                to_lower=True,
                denotation=test_weak_instruments_1,
            ),
            Example(
                input="How weak are the instrumental variables bedwards, tank, and dummy in a regression on stinger",
                semantics=("test_weak_instruments", "stinger", dep_vars_2),
                to_lower=True,
                denotation=test_weak_instruments_2,
            ),
            Example(
                input="How relevant in the first stage are instrument candidates gdp, easy, auto, and matic in a regression on enterq",
                semantics=("test_weak_instruments", "enterq", dep_vars_3),
                to_lower=True,
                denotation=test_weak_instruments_3,
            ),
            Example(
                input="Please test the weakness of instrument candidates stinger, matic, and beast in a regression on bedwards",
                semantics=("test_weak_instruments", "bedwards", dep_vars_1),
                to_lower=True,
                denotation=test_weak_instruments_1,
            ),
            Example(
                input="How strong are the instrumental variables bedwards, tank, and dummy in a regression on stinger",
                semantics=("test_weak_instruments", "stinger", dep_vars_2),
                to_lower=True,
                denotation=test_weak_instruments_2,
            ),
            Example(
                input="Test the relevance in the first stage are instrument candidates gdp, easy, auto, and matic in a regression on enterq",
                semantics=("test_weak_instruments", "enterq", dep_vars_3),
                to_lower=True,
                denotation=test_weak_instruments_3,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # find_instruments: proper grammar
            # find_instruments_1 = db.find_instruments("tank", "enterq", "bedwards", dep_vars_1).denotation
            # find_instruments_2 = db.find_instruments("beast", "enterq", "stinger", dep_vars_2).denotation
            # find_instruments_3 = db.find_instruments("bedwards", "enterq", "stinger", dep_vars_3).denotation
            # dep_vars_1 = ["stinger", "matic", "beast"]
            # dep_vars_2 = ["bedwards", "tank", "dummy"]
            # dep_vars_3 = ["gdp", "easy", "auto", "matic"]
            Example(
                input="Which instrumental variable candidates among stinger, matic, and beast are valid in a regression of tank on enterq, with bedwards as a known valid instrument?",
                semantics=(
                    "find_instruments",
                    "tank",
                    "enterq",
                    "bedwards",
                    dep_vars_1,
                ),
                to_lower=True,
                denotation=find_instruments_1,
            ),
            Example(
                input="What are the best instruments among bedwards, tank, and dummy in a regression of tank on enterq, if we know that stinger is exogenous?",
                semantics=(
                    "find_instruments",
                    "beast",
                    "enterq",
                    "stinger",
                    dep_vars_2,
                ),
                to_lower=True,
                denotation=find_instruments_2,
            ),
            Example(
                input="Which instrumental variables are valid among gdp, easy, auto, and matic in a regression of bedwards on enterq, if we know that stinger is valid?",
                semantics=(
                    "find_instruments",
                    "bedwards",
                    "enterq",
                    "stinger",
                    dep_vars_3,
                ),
                to_lower=True,
                denotation=find_instruments_3,
            ),
            Example(
                input="Among stinger, matic, and beast, which instruments are valid in a regression of tank on enterq, with bedwards as a known valid instrument?",
                semantics=(
                    "find_instruments",
                    "tank",
                    "enterq",
                    "bedwards",
                    dep_vars_1,
                ),
                to_lower=True,
                denotation=find_instruments_1,
            ),
            Example(
                input="What are the most valid instruments among bedwards, tank, and dummy in a regression of tank on enterq, if we know that stinger is exogenous?",
                semantics=(
                    "find_instruments",
                    "beast",
                    "enterq",
                    "stinger",
                    dep_vars_2,
                ),
                to_lower=True,
                denotation=find_instruments_2,
            ),
            Example(
                input="Which instrumental variables are the best -- among gdp, easy, auto, and matic -- in a regression of bedwards on enterq, if we know that stinger is valid?",
                semantics=(
                    "find_instruments",
                    "bedwards",
                    "enterq",
                    "stinger",
                    dep_vars_3,
                ),
                to_lower=True,
                denotation=find_instruments_3,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # print_a_bunch_of_AR_shit: proper grammar
            # print_a_bunch_of_AR_shit_1 = db.print_a_bunch_of_AR_shit("stinger", "year", 4).denotation
            # print_a_bunch_of_AR_shit_2 = db.print_a_bunch_of_AR_shit("beast", "year", 7).denotation
            # print_a_bunch_of_AR_shit_3 = db.print_a_bunch_of_AR_shit("auto", "year", 3).denotation
            # print_a_bunch_of_AR_shit_4 = db.print_a_bunch_of_AR_shit("matic", "year", 11).denotation
            Example(
                input="Run a 4-lag univariate autoregression on stinger with year as the time variable",
                semantics=("print_a_bunch_of_AR_shit", "stinger", "year", 4),
                to_lower=True,
                denotation=print_a_bunch_of_AR_shit_1,
            ),
            Example(
                input="Run a 7 lag time series regression on beast with year as the time variable",
                semantics=("print_a_bunch_of_AR_shit", "beast", "year", 7),
                to_lower=True,
                denotation=print_a_bunch_of_AR_shit_2,
            ),
            Example(
                input="Autoregression on auto with 3 lags and with year as the time variable",
                semantics=("print_a_bunch_of_AR_shit", "auto", "year", 3),
                to_lower=True,
                denotation=print_a_bunch_of_AR_shit_3,
            ),
            Example(
                input="Time series regression on matic with 11 lags and with year as the time variable",
                semantics=("print_a_bunch_of_AR_shit", "matic", "year", 11),
                to_lower=True,
                denotation=print_a_bunch_of_AR_shit_4,
            ),
            Example(
                input="Time series regression with 4 lags on stinger with year as the time variable",
                semantics=("print_a_bunch_of_AR_shit", "stinger", "year", 4),
                to_lower=True,
                denotation=print_a_bunch_of_AR_shit_1,
            ),
            Example(
                input="AR regression with seven lags on beast with year as the time variable",
                semantics=("print_a_bunch_of_AR_shit", "beast", "year", 7),
                to_lower=True,
                denotation=print_a_bunch_of_AR_shit_2,
            ),
            Example(
                input="Run time series AR on auto with 3 lags and with year as the time variable",
                semantics=("print_a_bunch_of_AR_shit", "auto", "year", 3),
                to_lower=True,
                denotation=print_a_bunch_of_AR_shit_3,
            ),
            Example(
                input="Univariate autoreg on matic with eleven lags and with year as the time variable",
                semantics=("print_a_bunch_of_AR_shit", "matic", "year", 11),
                to_lower=True,
                denotation=print_a_bunch_of_AR_shit_4,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # AR_with_moving_average: proper grammar
            # --------------------------------------------------------------------------------------------------------------- #
            # augmented_dicky_fuller_test: proper grammar
            # dicky_fuller_1 = db.augmented_dicky_fuller_test("bedwards")
            # dicky_fuller_2 = db.augmented_dicky_fuller_test("stinger")
            # dicky_fuller_3 = db.augmented_dicky_fuller_test("gdp")
            # dicky_fuller_4 = db.augmented_dicky_fuller_test("tank")
            Example(
                input="run dicky-fuller on bedwards",
                semantics=("augmented_dicky_fuller_test", "bedwards"),
                to_lower=True,
                denotation=dicky_fuller_1,
            ),
            Example(
                input="run stationarity test on bedwards",
                semantics=("augmented_dicky_fuller_test", "bedwards"),
                to_lower=True,
                denotation=dicky_fuller_1,
            ),
            Example(
                input="is stinger stationary as a time series variable",
                semantics=("augmented_dicky_fuller_test", "stinger"),
                to_lower=True,
                denotation=dicky_fuller_2,
            ),
            Example(
                input="run a stationarity test on gdp",
                semantics=("augmented_dicky_fuller_test", "gdp"),
                to_lower=True,
                denotation=dicky_fuller_3,
            ),
            Example(
                input="run a unit root test on the variable tank",
                semantics=("augmented_dicky_fuller_test", "tank"),
                to_lower=True,
                denotation=dicky_fuller_4,
            ),
            Example(
                input="does the variable tank have a unit root",
                semantics=("augmented_dicky_fuller_test", "tank"),
                to_lower=True,
                denotation=dicky_fuller_4,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # summarize_VAR: proper grammar
            # summarize_VAR_1 = db.summarize_VAR(dep_vars_1, "year", 3).denotation
            # summarize_VAR_2 = db.summarize_VAR(dep_vars_2, "year", 2).denotation
            # summarize_VAR_3 = db.summarize_VAR(dep_vars_3, "year", 2).denotation
            # dep_vars_1 = ["stinger", "matic", "beast"]
            # dep_vars_2 = ["bedwards", "tank", "matic"]
            # dep_vars_3 = ["gdp", "easy", "auto", "matic"]
            Example(
                input="Run a three-lag vector autoregression with the variables stinger, matic, and beast, and with year as the time variable, please.",
                semantics=("summarize_VAR", dep_vars_1, "year", 3),
                to_lower=True,
                denotation=summarize_VAR_1,
            ),
            Example(
                input="Can you run a general 2-lag multivariate autoregression with the variables bedwards, tank, and matic, and with year as the time variable?",
                semantics=("summarize_VAR", dep_vars_2, "year", 2),
                to_lower=True,
                denotation=summarize_VAR_2,
            ),
            Example(
                input="With year as the time index, run VAR with the variables gdp, easy, auto, and matic with 2 lags.",
                semantics=("summarize_VAR", dep_vars_3, "year", 2),
                to_lower=True,
                denotation=summarize_VAR_3,
            ),
            Example(
                input="VAR through year with 3 lags with the variables stinger, matic, and beast, please.",
                semantics=("summarize_VAR", dep_vars_1, "year", 3),
                to_lower=True,
                denotation=summarize_VAR_1,
            ),
            Example(
                input="Two lag multivariate autoregression with year for time with the variables bedwards, tank, and matic?",
                semantics=("summarize_VAR", dep_vars_2, "year", 2),
                to_lower=True,
                denotation=summarize_VAR_2,
            ),
            Example(
                input="Using bedwards, tank, and matic, run VAR through year with 2 lags.",
                semantics=("summarize_VAR", dep_vars_3, "year", 2),
                to_lower=True,
                denotation=summarize_VAR_3,
            ),

            # --------------------------------------------------------------------------------------------------------------- #
            # granger_causality_test: proper grammar
            # granger_causality_test_1 = db.granger_causality_test(dep_vars_1, "year", 3, "stinger").denotation
            # granger_causality_test_2 = db.granger_causality_test(dep_vars_2, "year", 3, "bedwards").denotation
            # granger_causality_test_3 = db.granger_causality_test(dep_vars_3, "year", 3, "gdp").denotation
            # dep_vars_1 = ["stinger", "matic", "beast"]
            # dep_vars_2 = ["bedwards", "tank", "matic"]
            # dep_vars_3 = ["gdp", "easy", "auto", "matic"]
            Example(
                input="Do matic and beast Granger cause stinger in a 3-lag time series regression with year as the time variable?",
                semantics=("granger_causality_test", dep_vars_1, "year", 3, "stinger",),
                to_lower=True,
                denotation=granger_causality_test_1,
            ),
            Example(
                input="Find whether dummy and tank Granger cause bedwards in a 3-lag time series regression with year as the time variable.",
                semantics=(
                    "granger_causality_test",
                    dep_vars_2,
                    "year",
                    3,
                    "bedwards",
                ),
                to_lower=True,
                denotation=granger_causality_test_2,
            ),
            Example(
                input="Run VAR with variables gdp, easy, auto, and matic, and then test whether gdp is Granger caused by the others.",
                semantics=("granger_causality_test", dep_vars_3, "year", 3, "gdp",),
                to_lower=True,
                denotation=granger_causality_test_3,
            ),
            Example(
                input="Find whether matic and beast eGranger cause stinger in a three lag regression with year as the time variable?",
                semantics=("granger_causality_test", dep_vars_1, "year", 3, "stinger",),
                to_lower=True,
                denotation=granger_causality_test_1,
            ),
            Example(
                input="Test whether dummy and tank exhibit Granger causality on bedwards in a three lag time series regression with year as the time variable.",
                semantics=(
                    "granger_causality_test",
                    dep_vars_2,
                    "year",
                    3,
                    "bedwards",
                ),
                to_lower=True,
                denotation=granger_causality_test_2,
            ),
            Example(
                input="Run a vector autoregression (time series) with variables gdp, easy, auto, and matic, and then test whether gdp is Granger caused by the others.",
                semantics=("granger_causality_test", dep_vars_3, "year", 3, "gdp",),
                to_lower=True,
                denotation=granger_causality_test_3,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # analyze_lags: proper grammar
            # analyze_lags_1 = db.analyze_lags("matic", "year")
            # analyze_lags_2 = db.analyze_lags(dep_vars_2, "year", max_lag=3)
            # analyze_lags_3 = db.analyze_lags("stinger", "year")
            #         dep_vars_2 = ["bedwards", "tank", "dummy"]
            Example(
                input="how many lags should I use in a time series regression through year on matic?",
                semantics=("analyze_lags", "matic", "year"),
                to_lower=True,
                denotation=analyze_lags_1,
            ),
            Example(
                input="how many lags should I use in a time series regression on bedwards, tank, and dummy, using year as the time?",
                semantics=("analyze_lags", dep_vars_2, "year"),
                to_lower=True,
                denotation=analyze_lags_2,
            ),
            Example(
                input="how many lagged values in autoregression on stinger that uses year as the time?",
                semantics=("analyze_lags", "stinger", "year"),
                to_lower=True,
                denotation=analyze_lags_3,
            ),
            Example(
                input="analyze the lags of a time series regression through year on matic?",
                semantics=("analyze_lags", "matic", "year"),
                to_lower=True,
                denotation=analyze_lags_1,
            ),
            Example(
                input="how many lagged values should I use in a regression on bedwards, tank, and dummy, using year as the time?",
                semantics=("analyze_lags", dep_vars_2, "year"),
                to_lower=True,
                denotation=analyze_lags_2,
            ),
            Example(
                input="find the lags in autoregression on stinger that uses year as the time?",
                semantics=("analyze_lags", "stinger", "year"),
                to_lower=True,
                denotation=analyze_lags_3,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # find_optimal_lag_length: proper grammar
            # find_optimal_lag_length_1 = db.find_optimal_lag_length("auto", "year").denotation
            # find_optimal_lag_length_2 = db.find_optimal_lag_length("matic", "year").denotation
            # find_optimal_lag_length_3 = db.find_optimal_lag_length(["bedwards", "tank"], "year").denotation
            # find_optimal_lag_length_4 = db.find_optimal_lag_length(["gdp", "easy"], "year").denotation
            # Example(
            #     input="What is the optimal number of lags in a time series regression on auto through year",
            #     semantics=("find_optimal_lag_length", "auto", "year"),
            #     to_lower=True,
            #     denotation=find_optimal_lag_length_1,
            # ),
            # Example(
            #     input="How many lags in autoregression on matic with year as the time",
            #     semantics=("find_optimal_lag_length", "matic", "year"),
            #     to_lower=True,
            #     denotation=find_optimal_lag_length_2,
            # ),
            # Example(
            #     input="What is the best number of lags in a vector autoregression on bedwards and tank with year as the time",
            #     semantics=("find_optimal_lag_length", ["bedwards", "tank"], "year"),
            #     to_lower=True,
            #     denotation=find_optimal_lag_length_3,
            # ),
            # Example(
            #     input="How many lags in VAR on gdp and easy with year as the time",
            #     semantics=("find_optimal_lag_length", ["gdp", "easy"], "year"),
            #     to_lower=True,
            #     denotation=find_optimal_lag_length_4,
            # ),
            # Example(
            #     input="Best number of lags in a time series regression on auto through year",
            #     semantics=("find_optimal_lag_length", "auto", "year"),
            #     to_lower=True,
            #     denotation=find_optimal_lag_length_1,
            # ),
            # Example(
            #     input="Optimal number of lags in autoregression on matic with year as the time",
            #     semantics=("find_optimal_lag_length", "matic", "year"),
            #     to_lower=True,
            #     denotation=find_optimal_lag_length_2,
            # ),
            # Example(
            #     input="Find optimal number of lags in a vector autoregression on bedwards and tank with year as the time",
            #     semantics=("find_optimal_lag_length", ["bedwards", "tank"], "year"),
            #     to_lower=True,
            #     denotation=find_optimal_lag_length_3,
            # ),
            # Example(
            #     input="Find the best number of lags in VAR on gdp and easy with year as the time",
            #     semantics=("find_optimal_lag_length", ["gdp", "easy"], "year"),
            #     to_lower=True,
            #     denotation=find_optimal_lag_length_4,
            # ),
            # --------------------------------------------------------------------------------------------------------------- #
            # print_PCA_wrapper: proper grammar
            # print_PCA_wrapper_1 = db.print_PCA_wrapper(["stinger", "beast"]).denotation
            # print_PCA_wrapper_2 = db.print_PCA_wrapper(["auto", "matic"]).denotation
            # print_PCA_wrapper_3 = db.print_PCA_wrapper(["dummy", "enterq"]).denotation
            Example(
                input="Run Principle Component Analysis on stinger and beast.",
                semantics=("print_PCA_wrapper", ["stinger", "beast"]),
                to_lower=True,
                denotation=print_PCA_wrapper_1,
            ),
            Example(
                input="Show the results of PCA on auto and matic.",
                semantics=("print_PCA_wrapper", ["auto", "matic"]),
                to_lower=True,
                denotation=print_PCA_wrapper_2,
            ),
            Example(
                input="Show the results of Principle Component Analysis on dummy and enterq.",
                semantics=("print_PCA_wrapper", ["dummy", "enterq"]),
                to_lower=True,
                denotation=print_PCA_wrapper_3,
            ),
            Example(
                input="Show the results of Principle Component Analysis on stinger and beast.",
                semantics=("print_PCA_wrapper", ["stinger", "beast"]),
                to_lower=True,
                denotation=print_PCA_wrapper_1,
            ),
            Example(
                input="Using auto and matic, show the results of PCA.",
                semantics=("print_PCA_wrapper", ["auto", "matic"]),
                to_lower=True,
                denotation=print_PCA_wrapper_2,
            ),
            Example(
                input="With dummy and enterq as the column variables, show the results of Principle Component Analysis.",
                semantics=("print_PCA_wrapper", ["dummy", "enterq"]),
                to_lower=True,
                denotation=print_PCA_wrapper_3,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # poisson_regression: proper grammar
            # poisson_1 = db.poisson_regression("gdp", "stinger")
            # poisson_2 = db.poisson_regression("auto", "matic")
            # poisson_3 = db.poisson_regression("bedwards", "gdp")
            # poisson_4 = db.poisson_regression("tank", "stinger")
            Example(
                input="Run a Poisson regression of gdp on stinger",
                semantics=("poisson_regression", "gdp", "stinger"),
                to_lower=True,
                denotation=poisson_1,
            ),
            Example(
                input="Poisson regress of auto on matic",
                semantics=("poisson_regression", "auto", "matic"),
                to_lower=True,
                denotation=poisson_2,
            ),
            Example(
                input="Regress bedwards on gdp asssuming bedwards is distributed Poisson",
                semantics=("poisson_regression", "bedwards", "gdp"),
                to_lower=True,
                denotation=poisson_3,
            ),
            Example(
                input="Run a Poisson regression of tank on stinger!",
                semantics=("poisson_regression", "tank", "stinger"),
                to_lower=True,
                denotation=poisson_4,
            ),
            Example(
                input="Show results of a Poisson regression of gdp on stinger",
                semantics=("poisson_regression", "gdp", "stinger"),
                to_lower=True,
                denotation=poisson_1,
            ),
            Example(
                input="What are the results if you Poisson regress of auto on matic",
                semantics=("poisson_regression", "auto", "matic"),
                to_lower=True,
                denotation=poisson_2,
            ),
            Example(
                input="Can you Poisson regress bedwards on gdp?",
                semantics=("poisson_regression", "bedwards", "gdp"),
                to_lower=True,
                denotation=poisson_3,
            ),
            Example(
                input="Please regress tank on stinger, assuming tank is distributed as a Poisson variable!",
                semantics=("poisson_regression", "tank", "stinger"),
                to_lower=True,
                denotation=poisson_4,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # markov_switching_regime_regression: proper grammar
            # markov_switching_regime_regression_1 = db.markov_switching_regime_regression("stinger", 3)
            # markov_switching_regime_regression_2 = db.markov_switching_regime_regression("gdp", 4)
            # markov_switching_regime_regression_3 = db.markov_switching_regime_regression("beast", 2)
            # markov_switching_regime_regression_4 = db.markov_switching_regime_regression("auto", 5)
            Example(
                input="Markov regression on Stinger with 3 states",
                semantics=("markov_switching_regime_regression", "stinger", 3),
                to_lower=True,
                denotation=markov_switching_regime_regression_1,
            ),
            Example(
                input="Markov regression on GDP with 4 regimes",
                semantics=("markov_switching_regime_regression", "gdp", 4),
                to_lower=True,
                denotation=markov_switching_regime_regression_2,
            ),
            Example(
                input="Run a Markov regression on Beast that has 2 regimes",
                semantics=("markov_switching_regime_regression", "beast", 2),
                to_lower=True,
                denotation=markov_switching_regime_regression_3,
            ),
            Example(
                input="With 5 regimes, run a Markov regression on auto.",
                semantics=("markov_switching_regime_regression", "auto", 5),
                to_lower=True,
                denotation=markov_switching_regime_regression_4,
            ),
            Example(
                input="Carry out markov regression, with 3 states, on Stinger",
                semantics=("markov_switching_regime_regression", "stinger", 3),
                to_lower=True,
                denotation=markov_switching_regime_regression_1,
            ),
            Example(
                input="Please run Markov regression on GDP with 4 regimes",
                semantics=("markov_switching_regime_regression", "gdp", 4),
                to_lower=True,
                denotation=markov_switching_regime_regression_2,
            ),
            Example(
                input="Can you Markov regress Beast with 2 regimes",
                semantics=("markov_switching_regime_regression", "beast", 2),
                to_lower=True,
                denotation=markov_switching_regime_regression_3,
            ),
            Example(
                input="Can you, with 5 regimes, run a Markov regression on auto.",
                semantics=("markov_switching_regime_regression", "auto", 5),
                to_lower=True,
                denotation=markov_switching_regime_regression_4,
            ),
            # --------------------------------------------------------------------------------------------------------------- #
            # choose_among_regression_models: proper grammar
            # np_list_1 = ["stinger", "matic", "beast"]
            # np_list_2 = ["bedwards", "tank", "dummy"]
            # np_list_3 = ["gdp", "easy", "auto", "matic"]
            # np_list_4 = ["easy", "matic", "enterq", "year", "animal"]
            # set_of_vars_1 = np.array([np_list_2, np_list_3, np_list_4])
            # set_of_vars_2 = np.array([np_list_1, np_list_3, np_list_4])
            # set_of_vars_3 = np.array([np_list_1, np_list_2, np_list_4])
            # set_of_vars_4 = np.array([np_list_1, np_list_2, np_list_3])
            # choose_among_regression_models_1 = db.choose_among_regression_models(set_of_vars_1).denotation
            # choose_among_regression_models_2 = db.choose_among_regression_models(set_of_vars_2).denotation
            # choose_among_regression_models_3 = db.choose_among_regression_models(set_of_vars_3).denotation
            # choose_among_regression_models_4 = db.choose_among_regression_models(set_of_vars_4).denotation
            Example(
                input="Which of these is the best model: bedwards on tank and dummy; gdp on easy, auto, and matic; or easy on matic, enterq, year, and animal",
                semantics=("choose_among_regression_models", set_of_vars_1),
                to_lower=True,
                denotation=choose_among_regression_models_1,
            ),
            Example(
                input="Which of these is the most accurate model: stinger on matic and beast; gdp on easy, auto, and matic; or easy on matic, enterq, year, and animal",
                semantics=("choose_among_regression_models", set_of_vars_2),
                to_lower=True,
                denotation=choose_among_regression_models_2,
            ),
            Example(
                input="Which of these is the best regressive model: stinger on matic and beast; bedwards on tank and dummy; or easy on matic, enterq, year, and animal",
                semantics=("choose_among_regression_models", set_of_vars_3),
                to_lower=True,
                denotation=choose_among_regression_models_3,
            ),
            Example(
                input="Which of these is the best model according to AIC: stinger on matic and beast; bedwards on tank and dummy; or gdp on easy, auto, and matic?",
                semantics=("choose_among_regression_models", set_of_vars_4),
                to_lower=True,
                denotation=choose_among_regression_models_4,
            ),
            Example(
                input="Find the optimal model in this list: bedwards on tank and dummy; gdp on easy, auto, and matic; or easy on matic, enterq, year, and animal",
                semantics=("choose_among_regression_models", set_of_vars_1),
                to_lower=True,
                denotation=choose_among_regression_models_1,
            ),
            Example(
                input="Find the best model here: stinger on matic and beast; gdp on easy, auto, and matic; or easy on matic, enterq, year, and animal",
                semantics=("choose_among_regression_models", set_of_vars_2),
                to_lower=True,
                denotation=choose_among_regression_models_2,
            ),
            Example(
                input="Find the most accurate model in this list: stinger on matic and beast; bedwards on tank and dummy; or easy on matic, enterq, year, and animal",
                semantics=("choose_among_regression_models", set_of_vars_3),
                to_lower=True,
                denotation=choose_among_regression_models_3,
            ),
            Example(
                input="According to Akaike Information Criterion, choose the truest model: stinger on matic and beast; bedwards on tank and dummy; or gdp on easy, auto, and matic?",
                semantics=("choose_among_regression_models", set_of_vars_4),
                to_lower=True,
                denotation=choose_among_regression_models_4,
            ),
        ]

        train_examples = denoted_examples + create_examples_from_amt(denoted_examples)
        return train_examples

    def get_examples_by_func(self, func):
        return [ex for ex in self.train_examples if ex.semantics[0] == func]
