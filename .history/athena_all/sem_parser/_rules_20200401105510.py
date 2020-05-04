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
    rules_query = [
        Rule(
            "$Query",
            "$Function ?$Optionals $Arguments",
            lambda sems: (sems[0], sems[2]),
        ),
        Rule(
            "$Query",
            "$Arguments ?$Optionals $Function",
            lambda sems: (sems[2], sems[0]),
        ),
    ]

    # ==================================================================================#
    # ARGUMENTS ========================================================================#
    # ==================================================================================#
    rules_arguments = [
        Rule("$Arguments", "$Argument", _sems_0),
        Rule("$Arguments", "$Arguments $Arguments", lambda sems: (sems[0], sems[1])),
        Rule(
            "$Argument",
            "$Argument ?$ArgJoin $Argument",
            lambda sems: (sems[0], sems[2]),
        ),
        Rule("$Argument", "$Column", _sems_0),
    ] + [Rule("$ArgJoin", joiner) for joiner in ["$Comma", "and"]]

    # ==================================================================================#
    # PUNCTUATION ======================================================================#
    # ==================================================================================#
    rules_punctuation = [
        Rule("$Comma", ","),
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
    # FUNCTIONS ========================================================================#
    # ==================================================================================#
    rules_functions = [
        Rule("$Function", "$" + func_name, lambda sems: sems[0])
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


### Function specific rules ###

