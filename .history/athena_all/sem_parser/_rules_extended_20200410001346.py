from athena_all.sem_parser.grammar import Rule
from athena_all.sem_parser._optionalwords import optional_words
from athena_all.databook import DataBook


def generate_extended_rules():

    var_words = ["variable", "var", "column"]
    hints_at_time = ["time", "index"]
    lag_words = ["lag", "-lag", "lags", "lagging", "lagged"]

    # First two are repeats, just to make the program run
    helper_rules = [
        Rule("$Column", word + "?$Optionals $Column", _sems_2) for word in var_words
    ]
    # + [Rule("$Column", "$Column ?$Optionals" + word, _sems_0) for word in var_words]

    # ==================================================================================#
    # TIME SERIES RULES ================================================================#
    # ==================================================================================#
    ###
    rules_general_time_series = [
        Rule("$TimeColumn", "$Column ?$Optionals $HintsAtTime", _sems_0),
        Rule("$Column", "$Column as the time variable", _sems_0),
        Rule("$Column", "$Column as the time index", _sems_0),
        Rule("$Column", "variable $Column", _sems_1),
    ]

    ###
    rules_print_a_bunch_of_AR_shit = (
        [
            Rule("$FunctionCall", "$print_a_bunch_of_AR_shitFunc", _sems_0),
            Rule(
                "$print_a_bunch_of_AR_shitFunc",
                "$print_a_bunch_of_AR_shit ?$Optionals $Column ?$Optionals $NumericalArgument ?$Optionals $Column",
                lambda sems: (sems[0], sems[2], sems[6], sems[4]),
            ),
            Rule(
                "$print_a_bunch_of_AR_shit",
                "$print_a_bunch_of_AR_shit ?Optionals $print_a_bunch_of_AR_shit",
                _sems_0,
            ),
            Rule("$NumericalArgument", "$NumericalArgument lags", _sems_0),
            Rule("$NumericalArgument", "lagging $NumericalArgument", _sems_1),
        ]
        + [Rule("$HintsAtTime", time_word, _sems_0) for time_word in hints_at_time]
        + [
            Rule("$NumericalArgument", lag + "?$Optionals $NumericalArgument", _sems_2)
            for lag in lag_words
        ]
    )

    ###
    rules_summarize_VAR = [
        Rule("$FunctionCall", "$summarize_VARFunc", _sems_0),
        Rule(
            "$summarize_VARFunc",
            "$summarize_VAR ?$Optionals $ColList ?$Optionals $NumericalArgument ?$Optionals $Column",
            lambda sems: (sems[0], sems[2], sems[6], sems[4]),
        ),
        Rule("$summarize_VAR", "$summarize_VAR ?Optionals $summarize_VAR", _sems_0),
    ]

    ###
    rules_augmented_dicky_fuller_test = [
        Rule("$FunctionCall", "$augmented_dicky_fuller_testFunc", _sems_0),
        Rule(
            "$augmented_dicky_fuller_testFunc",
            "$augmented_dicky_fuller_test ?$Optionals $Column",
            lambda sems: (sems[0], sems[2]),
        ),
        Rule(
            "$augmented_dicky_fuller_test",
            "$augmented_dicky_fuller_test ?Optionals $augmented_dicky_fuller_test",
            _sems_0,
        ),
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

    ### doesn't fucking work at all
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

    # ==================================================================================#
    # TITLE DIFFERENT RULE TYPES HERE===================================================#
    # ==================================================================================#

    return helper_rules


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
