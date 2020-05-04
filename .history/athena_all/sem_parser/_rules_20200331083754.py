from athena_all.sem_parrser.grammar import Rule
from athena_all.sem_parrser._optionalwords import optional_words


def rules(self):
    rules_optionals = [
        Rule("$ROOT", "?$Optionals $Query ?$Optionals", sems_1),
        Rule("$Optionals", "$Optional ?$Optionals"),
    ] + [Rule("$Optional", word) for word in optional_words]

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

    rules_arguments = [Rule("$Arguments", "$Column", sems_0)]

    rules_min_median_max = [
        Rule("$" + fnc, "$" + fnc + " $EntrySynonyms", sems_0)
        for fnc in ["findMin", "findMax", "findMedian"]
    ]

    rules_entry_syns = [
        Rule("$EntrySynonyms", syn) for syn in ["entry", "number", "value"]
    ]

    rules_functions = [
        Rule("$Function", "$" + func_name, lambda sems: sems[0])
        for func_name in self.db.get_function_names()
    ]

    return (
        rules_optionals
        + rules_query
        + rules_functions
        + rules_arguments
        + rules_min_median_max
        + rules_entry_syns
    )

