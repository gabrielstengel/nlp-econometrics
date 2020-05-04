from athena_all.sem_parser.grammar import Rule
from athena_all.sem_parser._optionalwords import optional_words
from athena_all.databook import DataBook


def generate_rules(db: DataBook):
    """ Generate the rules to be used in the grammar for pasing quries on this databook. """

    # ==================================================================================#
    # OPTIONALS ========================================================================#
    # ==================================================================================#
    rules_optionals = [
        Rule("$ROOT", "?$Optionals $Query ?$Optionals", _sems_1),
        Rule("$Optionals", "$Optional ?$Optionals"),
    ] + [Rule("$Optional", word) for word in optional_words]

    # ==================================================================================#
    # QUERY STRUCTURE ==================================================================#
    # ==================================================================================#
    rules_query = 
        [
            Rule(
                "$Query",
                "$Function ?$Optionals $Argument",
                lambda sems: (sems[0], sems[2]),
            ),
            Rule(
                "$Query",
                "$Argument ?$Optionals $Function",
                lambda sems: (sems[2], sems[0]),
            ),
            Rule("$Query", "?$Optionals $FunctionCall ?$Optionals", _sems_1),
            Rule(
                "$FunctionCall",
                "$SingleColArgFunc ?Optionals $Column",
                lambda sems: (sems[0], sems[2]),
            ),
            Rule(
                "$FunctionCall",
                "$Column ?Optionals $SingleColArgFunc",
                lambda sems: (sems[2], sems[0]),
            )
        ]
    #+ [Rule("$SingleColArgFunc", func, _sems_0,) for func in ['$findMean','$findMedian', '$findStd', ]]
    

    # ==================================================================================#
    # ARGUMENTS ========================================================================#
    # ==================================================================================#

    column_arguments = [
        Rule("$Argument", "$Column", _sems_0),
        Rule("$ColList", "$Column ?$ColJoin $Column", lambda sems: [sems[0], sems[2]]),
        Rule(
            "$ColList",
            "$Column ?$ColJoin $ColList",
            lambda sems: [sem for sem in [sems[0]] + sems[2]],
        ),
        Rule("$ColJoin", "$Comma ?$And"),
        Rule("$ColJoin", "$And"),
    ]

    rules_arguments = [
        Rule("$Arguments", "$Arguments $Arguments", lambda sems: (sems[0], sems[1])),
        Rule(
            "$Argument",
            "$Argument ?$ArgJoin $Argument",
            lambda sems: (sems[0], sems[2]),
        ),
    ]
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

    # ==================================================================================#
    # PUNCTUATION AND TRANSITION WORDS =================================================#
    # ==================================================================================#
    rules_punctuation = [
        Rule("$Comma", ","),
        Rule("$And", "and"),
        Rule("$As", "as"),
        Rule("$QuestionMark", "?"),
    ]

    # ==================================================================================#
    # MIN MEDIAN AND MAX ===============================================================#
    # ==================================================================================#
    rules_min_median_max = [
        Rule("$" + fnc, "$" + fnc + " $EntrySynonyms", _sems_0)
        for fnc in ["findMin", "findMax", "findMedian"]
    ]

    rules_entry_syns = [
        Rule("$EntrySynonyms", syn) for syn in ["entry", "number", "value"]
    ]

    # ==================================================================================#
    # LOOK AT COLUMNS ==================================================================#
    # ==================================================================================#
    rules_look_at_data = [
        Rule(
            "$FunctionCall",
            phrase + " ?column $Column",
            lambda sems: ("showCol", sems[-1]),
        )
        for phrase in [
            "show me",
            "let me see",
            "print",
            "print",
            "what is",
            "display",
            "lets see",
            "look at",
            "give me",
        ]
    ]

    # ==================================================================================#
    # GREETINGS ========================================================================#
    # ==================================================================================#
    rules_greetings = [Rule("$FunctionCall", "how are you", lambda sems: ("greeting"))]

    # ==================================================================================#
    # FUNCTIONS ========================================================================#
    # ==================================================================================#
    rules_functions = [
        Rule("$Function", "$" + func_name, lambda sems: sems[0])
        for func_name in db.get_function_names()
        if not func_name == "multiReg"
    ]

    return (
        rules_optionals
        + rules_query
        + rules_functions
        + rules_arguments
        + rules_min_median_max
        + rules_entry_syns
        + rules_punctuation
        + column_arguments
        + rules_multi_regression
        + rules_ivRegress
        + rules_look_at_data
        + rules_greetings
    )


# semantics helper functions ===========================================================
# for handling the semantics (i.e. building them during rule parsing)
def _sems_0(sems):
    return sems[0]


def _sems_1(sems):
    return sems[1]


def reverse(relation_sem):
    """TODO"""
    # relation_sem is a lambda function which takes an arg and forms a pair,
    # either (rel, arg) or (arg, rel).  We want to swap the order of the pair.
    def apply_and_swap(arg):
        pair = relation_sem(arg)
        return (pair[1], pair[0])

    return apply_and_swap


#  =====================================================================================


### For creating rules where the order shouldn't matter. ###


def create_any_order(rule):
    """ NEEEDS TO BE IMPLEMENTED """
    return []

