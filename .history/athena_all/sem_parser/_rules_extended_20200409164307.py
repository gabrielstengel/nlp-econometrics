from athena_all.sem_parser.grammar import Rule
from athena_all.sem_parser._optionalwords import optional_words
from athena_all.databook import DataBook


def generate_extended_rules():

    var_words = ["variable", "var", "column"]
    hints_at_time = ["time", "index"]

    # First two are repeats, just to make the program run
    helper_rules = [
        Rule("$Column", word + "?$Optionals $Column", _sems_2) for word in var_words
    ]
    # + [Rule("$Column", "$Column ?$Optionals" + word, _sems_0) for word in var_words]

    # ==================================================================================#
    # TIME SERIES RULES ================================================================#
    # ==================================================================================#

    general_time_series_rules = [
        Rule("$TimeColumn", "$Column ?$Optionals $HintsAtTime", _sems_0),
    ]

    print_a_bunch_of_AR_shit_rules = [
        Rule("$FunctionCall", "$print_a_bunch_of_AR_shitFunc", _sems_0),
        Rule(
            "$print_a_bunch_of_AR_shitFunc",
            "?$Optionals $print_a_bunch_of_AR_shit ?$Optionals $Column ?$Optionals $time_var_plus_lags ?$Optionals",
            lambda sems: (sems[1], sems[3], sems[5]),
        ),
        Rule(
            "$time_var_plus_lags",
            "$Column ?$Optionals $Argument ?$Optionals",
            lambda sems: (sems[0], sems[2]),
        ),
        Rule(
            "$time_var_plus_lags",
            "$Argument ?$Optionals $TimeColumn ?$Optionals",
            lambda sems: (sems[2], sems[0]),
        ),
        # Rule(
        #     "$print_a_bunch_of_AR_shitFunc",
        #     "?$Optionals $print_a_bunch_of_AR_shit ?$Optionals $Column ?$Optionals $Column ?$Optionals",
        #     lambda sems: (sems[1], sems[3], sems[5]),
        # ),
        # Rule(
        #     "$print_a_bunch_of_AR_shitFunc",
        #     "?$Optionals $print_a_bunch_of_AR_shit ?$Optionals $Column ?$Optionals $Argument ?$Optionals $TimeColumn ?$Optionals ",
        #     lambda sems: (sems[1], sems[3], sems[7], sems[5]),
        # ),
        Rule("$TimeColumn", "$Column ?$Optionals $HintsAtTime", _sems_0),
    ] + [Rule("$HintsAtTime", time_word, _sems_0) for time_word in hints_at_time]

    augmented_dicky_fuller_test_rules = [
        Rule("$Column", "variable $Column", _sems_1),
        Rule("$FunctionCall", "$augmented_dicky_fuller_testFunc", _sems_0),
        Rule(
            "$augmented_dicky_fuller_testFunc",
            "?$Optionals $augmented_dicky_fuller_test ?$Optionals $Column ?$Optionals",
            lambda sems: (sems[1], sems[3]),
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

    # ==================================================================================#
    # TITLE DIFFERENT RULE TYPES HERE===================================================#
    # ==================================================================================#

    return (
        helper_rules
        + general_time_series_rules
        + print_a_bunch_of_AR_shit_rules
        + augmented_dicky_fuller_test_rules
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
