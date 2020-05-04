from athena_all.sem_parser.grammar import Rule
from athena_all.sem_parser._optionalwords import optional_words
from athena_all.databook import DataBook


def generate_extended_rules():

    var_words = ["variable", "var", "column", "regressor", "covariate", "index", "dimension"]
    hints_at_time = ["time", "index", "time series"]
    hints_at_id = ["id", "identification", "person", "identifier"]
    lag_words = ["lag", "-lag", "lags", "lagging", "lagged"]
    instrument_syns = [
        "instruments",
        "instrument vars",
        "instrument variables",
        "instrumental variables",
    ]
    reg_functions = [
        "$reg",
        "$multiReg",
        "$fixedEffects",
        "$summarizeLogisticRegression",
        "$logisticMarginalEffects",
        "$ivRegress",
        "$homoskedasticJStatistic",
        "$find_instruments",
        "$print_a_bunch_of_AR_shit",
        "$AR_with_moving_average",
        "$summarize_VAR",
        "$granger_causality_test",
        "$print_PCA_wrapper",
        "$poisson_regression",
        "$markov_switching_regime_regression",
    ]

    rules_random = [
        Rule("$The", "the"),
        Rule("$A", "a"),
        Rule("$For", "for"),
    ] + [Rule("$Column", word + "?$Optionals $Column", _sems_2) for word in var_words]

    # ==================================================================================#
    # RULES FOR DISTINGUISHING VARIABLE TYPES ==========================================#
    # ==================================================================================#

    rules_iv_vars = [
        Rule("$Instruments", "$ColList", _sems_0),
        Rule("$Instruments", "$ColList ?$As $InstrumentSyns", _sems_0, add_all_permutations=True),
        Rule("$Instruments", "$Column", _sems_0),
        Rule("$Instruments", "$Column ?$As $InstrumentSyns", _sems_0, add_all_permutations=True),
        Rule("$Instruments", "$Instruments $VarWord", _sems_0, add_all_permutations=True),
    ] + [Rule("$InstrumentSyns", ins_syns) for ins_syns in instrument_syns]

    rules_dep_vars = [
        Rule("$DepVar", "of $Column", _sems_1),
        Rule("$DepVar", "$Column ?$As ?$The ?$A dependent", _sems_0, add_all_permutations=True),
        Rule("$DepVar", "$regress_function_word $Column", _sems_1),
        Rule("$DepVar", "$DepVar $VarWord", _sems_0, add_all_permutations=True),
    ] + [Rule("$regress_function_word", reg_f) for reg_f in reg_functions]

    rules_ind_vars= [
        Rule("$IndVar", "on $Column", _sems_1),
        Rule("$IndVar", "using $Column", _sems_1),
        Rule("$IndVar", "varies with $Column", _sems_1),
        Rule("$IndVar", "changes with $Column", _sems_1),
        Rule("$IndVar", "$Column ?$As ?$The ?$A independent", _sems_0, add_all_permutations=True),
        Rule("$IndVarList", "$IndVar ?$ColJoin $IndVar", lambda sems: [sems[0], sems[2]]),
        Rule(
            "$IndVarList",
            "$IndVar ?$ColJoin $ColList",
            lambda sems: [sem for sem in [sems[0]] + sems[2]],
        ),
        Rule("$IndVar", "$IndVar $VarWord", _sems_0, add_all_permutations=True),
    ]

    rule_time_vars = [
        Rule("$TimeColumn", "$Column time", _sems_0, add_all_permutations=True, add_optionals_btw=True),
        Rule("$TimeColumn", "$Column ?As ?The ?For $HintsAtTime", _sems_0),
        Rule("$TimeColumn", "$TimeColumn $VarWord", _sems_0, add_all_permutations=True),
    ] + [Rule("$HintsAtTime", time_word, _sems_0) for time_word in hints_at_time
    ] + [Rule("$VarWord", var_word, _sems_0) for var_word in var_words]

    rules_id_vars = [
        Rule("$IDColumn", "$Column $HintsAtID", _sems_0, add_all_permutations=True, add_optionals_btw=True),
        Rule("$IDColumn", "$IDColumn $VarWord", _sems_0, add_all_permutations=True),
    ] + [Rule("$HintsAtID", id_word, _sems_0) for id_word in hints_at_id]

    rules_model = [
        Rule("$Model", "$DepVar $IndVar", lambda sems: [sems[0], sems[1]], add_all_permutations=True),
        Rule("$Model", "$DepVar $IndVarist", lambda sems: [sems[0], sems[1]], add_all_permutations=True),
    ]

    # # getting rid of these asap
    # rules_to_be_removed = [
    #     # Rule("$Column", "$Instruments", _sems_0),
    #     Rule("$Column", "$DepVar", _sems_0),
    #     Rule("$Column", "$IndVar", _sems_0),
    #     Rule("$Column", "$TimeColumn", _sems_0),
    # ]

    # ==================================================================================#
    # TIME SERIES RULES ================================================================#
    # ==================================================================================#

    ###===###
    rules_general_time_series = [
        Rule("$Column", "$Column index", _sems_0),
        Rule("$Column", "variable $Column", _sems_1, add_all_permutations=True, add_optionals_btw=True),
        Rule("$Column", "variables $Column", _sems_1, add_all_permutations=True, add_optionals_btw=True),
        Rule("$Column", "column $Column", _sems_1, add_all_permutations=True, add_optionals_btw=True),
        Rule("$Column", "columns $Column", _sems_1, add_all_permutations=True, add_optionals_btw=True),
        Rule("$Column", "using $Column", _sems_1, add_all_permutations=True, add_optionals_btw=True),
        Rule("$NumericalArgument", "$NumericalArgument lags", _sems_0, add_all_permutations=True, add_optionals_btw=True),
        Rule("$NumericalArgument", "$NumericalArgument lagged", _sems_0, add_all_permutations=True, add_optionals_btw=True),
        Rule("$NumericalArgument", "$NumericalArgument values", _sems_0, add_all_permutations=True, add_optionals_btw=True),
        Rule("$NumericalArgument", "lagging $NumericalArgument", _sems_1, add_all_permutations=True, add_optionals_btw=True),
        Rule("$NumericalArgument", "lag $NumericalArgument", _sems_1, add_all_permutations=True, add_optionals_btw=True),
        Rule("$Optional", "$Optional variables", _sems_0),
        Rule("$Optional", "$Optional variable", _sems_0),
    ]

    ###===###
    rules_print_a_bunch_of_AR_shit = [
        Rule("$FunctionCall", "$print_a_bunch_of_AR_shitFunc", _sems_0),
        Rule(
            "$print_a_bunch_of_AR_shitFunc",
            "$print_a_bunch_of_AR_shit $Column $NumericalArgument $TimeColumn",
            lambda sems: (sems[0], sems[1], sems[3], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True
        ),
        Rule("$print_a_bunch_of_AR_shit", "$print_a_bunch_of_AR_shit ?Optionals $print_a_bunch_of_AR_shit", _sems_0),
    ] + [Rule("$NumericalArgument", lag + "?$Optionals $NumericalArgument", _sems_2) for lag in lag_words]


    ###===###
    rules_summarize_VAR = [
        Rule("$FunctionCall", "$summarize_VARFunc", _sems_0),
        Rule(
            "$summarize_VARFunc",
            "$summarize_VAR $ColList $NumericalArgument $TimeColumn",
            lambda sems: (sems[0], sems[1], sems[3], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True
        ),
        Rule("$summarize_VAR", "$summarize_VAR ?Optionals $summarize_VAR", _sems_0),
    ]

    ###===###
    rules_augmented_dicky_fuller_test = [
        Rule("$FunctionCall", "$augmented_dicky_fuller_testFunc", _sems_0),
        Rule(
            "$augmented_dicky_fuller_testFunc",
            "$augmented_dicky_fuller_test ?$Optionals $Column",
            lambda sems: (sems[0], sems[2]),
        ),
        Rule("$augmented_dicky_fuller_test",
            "$augmented_dicky_fuller_test ?Optionals $augmented_dicky_fuller_test", 
            _sems_0),
        Rule("$augmented_dicky_fuller_test",
            "$augmented_dicky_fuller_test variable", 
            _sems_0),
        Rule("$augmented_dicky_fuller_test",
            "$augmented_dicky_fuller_test as a time series variable", 
            _sems_0),
        Rule(
            "$augmented_dicky_fuller_testFunc",
            "?$Optionals  $Column ?$Optionals $augmented_dicky_fuller_test ?$Optionals",
            lambda sems: (sems[3], sems[1]),
        ),
        Rule(
            "$augmented_dicky_fuller_test",
            "?$Optionals $augmented_dicky_fuller_test ?$Optionals $augmented_dicky_fuller_test ?$Optionals",
            _sems_1,
        ),
    ]

    ###===### atrocious at the moment
    rules_granger_causality_test = [
        Rule("$FunctionCall", "$granger_causality_testFunc", _sems_0),
        Rule(
            "$granger_causality_testFunc",
            "$ColList ?$Optionals $granger_causality_test ?$Optionals $Argument ?$Optionals $Column ?$Optionals $Column ?$Optionals $ColList",
            lambda sems: (sems[2], sems[0], sems[6], sems[4], sems[8], sems[10]),
        ),
        Rule(
            "$granger_causality_test",
            "$granger_causality_test ?$Optionals $granger_causality_test",
            _sems_0,
        ),
    ]


    ###===### 
    rules_analyze_lags = [
        Rule("$FunctionCall", "$analyze_lagsFunc", _sems_0),
        Rule(
            "$analyze_lagsFunc",
            "$analyze_lags $ColList $TimeColumn",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True
            ),
        Rule(
            "$analyze_lagsFunc",
            "$analyze_lags $Column $TimeColumn",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True
            ),
        Rule("$analyze_lags", "$analyze_lags ?$Optionals $analyze_lags", _sems_0),
    ]


    # ==================================================================================#
    # ADVANCED REGRESSION METHOD RULES =================================================#
    # ==================================================================================#
    ###===###
    rules_general_advanced_regression_methods = [
        Rule("$ColPair", "$Column ?$ColJoin $Column", lambda sems: [sems[0], sems[2]]),
        Rule("$NumericalArgument", "regimes $NumericalArgument", _sems_1, add_all_permutations=True, add_optionals_btw=True),
        Rule("$NumericalArgument", "states $NumericalArgument", _sems_1, add_all_permutations=True, add_optionals_btw=True),
        Rule("$NumericalArgument", "regime $NumericalArgument", _sems_1, add_all_permutations=True, add_optionals_btw=True),
        Rule("$NumericalArgument", "state $NumericalArgument", _sems_1, add_all_permutations=True, add_optionals_btw=True),

        # obviously need to find a way for it not to be a semicolon
        Rule("$SetsOfColLists", "$ColList ; $ColList", lambda sems: [sems[0], sems[2]]),
        Rule("$SetsOfColLists", "$SetsOfColLists ; $ColList", lambda sems: [sems[0], sems[2]]),
        Rule("$SetsOfColLists", "$ColList or $ColList", lambda sems: [sems[0], sems[2]]),
        Rule("$SetsOfColLists", "$SetsOfColLists or $ColList", lambda sems: [sems[0], sems[2]]),
    ]

    ###===###
    rules_print_PCA_wrapper = [
        Rule("$FunctionCall", "$print_PCA_wrapperFunc", _sems_0),
        Rule(
            "$print_PCA_wrapperFunc",
            "$print_PCA_wrapper $ColPair",
            lambda sems: (sems[0], sems[1]),
            add_all_permutations=True,
            add_optionals_btw=True
            ),
        Rule("$print_PCA_wrapper", "$print_PCA_wrapper ?$Optionals $print_PCA_wrapper", _sems_0),
    ]

    ###===###
    rules_poisson_regression = [
        Rule("$FunctionCall", "$poisson_regressionFunc", _sems_0),
        Rule(
            "$poisson_regressionFunc",
            "$poisson_regression $DepVar $IndVar",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True
        ),
        Rule("$poisson_regression", "$poisson_regression ?$Optionals $poisson_regression", _sems_0),
    ]

    ###===###
    rules_markov_switching_regime_regression = [
        Rule("$FunctionCall", "$markov_switching_regime_regressionFunc", _sems_0),
        Rule(
            "$markov_switching_regime_regressionFunc",
            "$markov_switching_regime_regression $Column $NumericalArgument",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True
        ),
        Rule(
            "$markov_switching_regime_regression",
            "$markov_switching_regime_regression ?$Optionals $markov_switching_regime_regression",
            _sems_0
        ),
    ]

    ###===### Really shitty bc it doesn't differentiate between dep and ind. Should maybe just do $Column $IndVar
    rules_choose_among_regression_models = [
        Rule("$FunctionCall", "$choose_among_regression_modelsFunc", _sems_0),
        Rule(
            "$choose_among_regression_modelsFunc",
            "$choose_among_regression_models $SetsOfColLists",
            lambda sems: (sems[0], sems[1]),
            add_all_permutations=True,
            add_optionals_btw=True
        ),
        Rule(
            "$choose_among_regression_models",
            "$choose_among_regression_models ?$Optionals $choose_among_regression_models",
            _sems_0
        ),
        Rule(
            "$choose_among_regression_models",
            "$choose_among_regression_models from",
            _sems_0
        ),
        Rule(
            "$choose_among_regression_models",
            "$choose_among_regression_models among these",
            _sems_0
        ),
        Rule(
            "$choose_among_regression_models",
            "$choose_among_regression_models in this list",
            _sems_0
        ),
    ]









    # rules_ivRegress = [
    #     Rule("$FunctionCall", "$ivRegressFunc", _sems_0),
    #     Rule(
    #         "$ivRegressFunc",
    #         "$ivRegress $Column $ColList $Instruments",
    #         lambda sems: (sems[0], sems[1], sems[2], sems[3]),
    #         add_all_permutations=True,
    #         add_optionals_btw=True,
    #     ),
    #     Rule(
    #         "$ivRegressFunc",
    #         "$ivRegress $Column $Column $Instruments",
    #         lambda sems: (sems[0], sems[1], sems[2], sems[3]),
    #         add_all_permutations=True,
    #         add_optionals_btw=True,
    #     ),
    #     Rule("$ivRegress", "$ivRegress ?$Optionals $ivRegress", _sems_0),
    # ]

    # rules_homoskedasticJStatistic = [
    #     Rule("$FunctionCall", "$homoskedasticJStatisticFunc", _sems_0),
    #     Rule(
    #         "$homoskedasticJStatisticFunc",
    #         "$homoskedasticJStatistic $Column $ColList $Instruments",
    #         lambda sems: (sems[0], sems[1], sems[2], sems[3]),
    #         add_all_permutations=True,
    #         add_optionals_btw=True,
    #     ),
    #     Rule(
    #         "$homoskedasticJStatisticFunc",
    #         "$homoskedasticJStatistic $Column $Column $Instruments",
    #         lambda sems: (sems[0], sems[1], sems[2], sems[3]),
    #         add_all_permutations=True,
    #         add_optionals_btw=True,
    #     ),
    #     Rule(
    #         "$homoskedasticJStatistic",
    #         "$homoskedasticJStatistic $ivRegress",
    #         _sems_0,
    #         add_all_permutations=True,
    #         add_optionals_btw=True,
    #     ),
    #     Rule(
    #         "$homoskedasticJStatistic",
    #         "$homoskedasticJStatistic test",
    #         _sems_0,
    #         add_all_permutations=True,
    #         add_optionals_btw=True,
    #     ),
    #     Rule("$homoskedasticJStatistic", "$homoskedasticJStatistic ?$Optionals $homoskedasticJStatistic", _sems_0),
    # ]







    # ==================================================================================#
    # TITLE DIFFERENT RULE TYPES HERE===================================================#
    # ==================================================================================#

    return (
        rules_random
        + rules_general_time_series
        + rules_print_a_bunch_of_AR_shit
        + rules_augmented_dicky_fuller_test
        + rules_summarize_VAR
        + rules_granger_causality_test
        + rules_analyze_lags
        + rules_general_advanced_regression_methods
        + rules_print_PCA_wrapper
        + rules_poisson_regression
        + rules_markov_switching_regime_regression
        + rules_choose_among_regression_models
        + rules_iv_vars
        + rules_ind_vars
        + rules_dep_vars
        + rule_time_vars
        + rules_id_vars
        + rules_model
        # + rules_to_be_removed
    )


"""
    # ==================================================================================#
    # MULTI REGRESSION =================================================================#
    # ==================================================================================#
    rules_multi_regression = [
        Rule("$FunctionCall", "$multiRegFunc", _sems_0),
        Rule(
            "$multiRegFunc",
            "$multiReg ?$Optionals $Column ?$Optionals $ColList",
            lambda sems: (sems[0], sems[2], sems[4]),
        ),
        Rule(
            "$multiRegFunc",
            "$ColList ?Optionals $multiReg ?$Optionals $Column",
            lambda sems: (sems[2], sems[4], sems[0]),
        ),
    ]

    # ==================================================================================#
    # INSTRUMENTAL VARIABLE REGRESSION =================================================#
    # ==================================================================================#
    instrument_syns = [
        "instruments",
        "instrument vars",
        "instrument variables",
        "instrumental variables",
    ]

    rules_ivRegress = [
        Rule("$FunctionCall", "$ivRegressFunc", _sems_0),
        Rule(
            "$ivRegressFunc",
            "$ivRegress ?$Optionals $Column ?$Optionals $ColList $Instruments",
            lambda sems: (sems[0], sems[2], sems[4]),
        ),
        Rule("$Intruments", "$ColList"),
        Rule("$Intruments", "$ColList ?$As $InstrumentSyns"),
    ] + [Rule("$InstrumentSyns", ins_syns) for ins_syns in instrument_syns]

    """


# semantics helper functions ===========================================================
# for handling the semantics (i.e. building them during rule parsing)
def _sems_0(sems):
    return sems[0]


def _sems_1(sems):
    return sems[1]


def _sems_2(sems):
    return sems[2]
