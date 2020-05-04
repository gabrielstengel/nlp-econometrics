from athena_all.sem_parser.grammar import Rule
from athena_all.sem_parser._optionalwords import optional_words, optional_help_words
from athena_all.databook import DataBook


def generate_rules(db: DataBook):
    """ Generate the rules to be used in the grammar for pasing quries on this databook. """

    # ==================================================================================#
    # OPTIONALS ========================================================================#
    # ==================================================================================#
    rules_optionals = [
        Rule("$ROOT", "?$Optionals $Query ?$Optionals", _sems_1),
        Rule("$Optionals", "$Optional ?$Optionals"),
        # Rule("$Optional", "$Token"),  # 13.6% # 19.4% with this rule
    ] + [Rule("$Optional", word) for word in optional_words]

    # ==================================================================================#
    # QUERY STRUCTURE ==================================================================#
    # ==================================================================================#
    rules_query = [
        Rule("$Query", "$FunctionCall", _sems_0),
    ]

    # ==================================================================================#
    # QUERY STRUCTURE ==================================================================#
    # ==================================================================================#
    rules_single_col = [
        Rule(
            "$FunctionCall",
            "$SingleColArgFunc ?Optionals $Column",
            lambda sems: (sems[0], sems[2]),
            add_all_permutations=True,
        ),
    ] + [
        Rule("$SingleColArgFunc", func, _sems_0,)
        for func in [
            "$findMean",
            "$findMedian",
            "$findStd",
            "$findVar",
            "$findMax",
            "$findMin",
        ]
    ]

    # ==================================================================================#
    # ARGUMENTS ========================================================================#
    # ==================================================================================#

    column_arguments = [
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
        Rule("$Arguement", "$NumericalArgument", _sems_0),
        Rule("$NumericalArgument", "$Number", _sems_0),
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
        Rule("$EntrySynonyms", syn) for syn in ["entry", "number", "value", "variable"]
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
    rules_greetings = [
        Rule("$FunctionCall", "how ?are you", ("greeting",)),
        Rule("$FunctionCall", "how's it going", ("greeting",)),
        Rule("$FunctionCall", "hows it going", ("greeting",)),
        Rule("$FunctionCall", "what’s up", ("greeting",)),
        Rule("$FunctionCall", "whats up", ("greeting",)),
        Rule("$FunctionCall", "what's up", ("greeting",)),
    ]

    # ==================================================================================#
    # HELP===============================================================================#
    # ==================================================================================#
    rules_help = [
        Rule("$FunctionCall", "$HelpFunction", _sems_0),
        Rule(
            "$HelpFunction",
            "$HelpWords ?$Optionals $Function",
            lambda sems: ("help", sems[2]),
        ),
        Rule(
            "$HelpFunction",
            "$Function ?$Optionals $HelpWords",
            lambda sems: ("help", sems[2]),
        ),
        Rule("$You", "you"),
        Rule("$Know", "?you know ?how"),
        Rule(
            "$HelpFunction",
            "$QuestionPhrase ?$Optionals $Function",
            lambda sems: ("help", sems[2]),
        ),
    ] + [
        Rule("$HelpWords", hw)
        for hw in [
            "help",
            "aid",
            "assist",
            "assistance",
            "helping",
            "aiding",
            "assisting",
            "assistance",
        ]
    ]

    # ==================================================================================#
    # QUESTION PHRASES =================================================================#
    # ==================================================================================#
    question_phrase_rules = [
        Rule("$What", "what ?is"),
        Rule("$What", "what's"),
        Rule("$What", "whats"),
        Rule("$How", "how do"),
        Rule("$How", "how does"),
        Rule("$How", "how to"),
        Rule("$How", "how do"),
        Rule("$Can", "can ?you"),
        Rule("$Can", "can i"),
        Rule("$Can", "could i"),
        Rule("$Can", "could ?you"),
        Rule("$Can", "would ?you"),
        Rule("$Can", "would i"),
        Rule("$Can", "will i"),
        Rule("$Can", "will ?you"),
    ] + [Rule("$QuestionPhrase", qp) for qp in ["$What", "$How", "$Can",]]

    # ==================================================================================#
    # FUNCTIONS ========================================================================#
    # ==================================================================================#
    rules_functions = [
        Rule("$Function", "$" + func_name, _sems_0)
        for func_name in db.get_function_names()
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
        + rules_help
        + rules_greetings
        + rules_single_col
        + question_phrase_rules
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

