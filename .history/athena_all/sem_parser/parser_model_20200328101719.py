from collections import defaultdict
from numbers import Number

import pandas as pd

from athena_all.sem_parser.traindata.econ_train_examples import (
    train_example_matrix,
    train_examples,
)
from athena_all.sem_parser.experiment import (
    evaluate_for_domain,
    evaluate_dev_examples_for_domain,
    train_test,
    train_test_for_domain,
    interact,
    learn_lexical_semantics,
    generate,
)

from athena_all.sem_parser.grammar import (
    Grammar,
    Rule,
    FunctionAnnotator,
    ColumnAnnotator,
    Domain,
    Example,
    DenotationAccuracyMetric,
    denotation_match_metrics,
    rule_features,
)

from athena_all.databook import DataBook

# EconometricSemanticParser ============================================================

# semantics helper functions ===========================================================


def sems_0(sems):
    return sems[0]


def sems_1(sems):
    return sems[1]


def reverse(relation_sem):
    """TODO"""
    # relation_sem is a lambda function which takes an arg and forms a pair,
    # either (rel, arg) or (arg, rel).  We want to swap the order of the pair.
    def apply_and_swap(arg):
        pair = relation_sem(arg)
        return (pair[1], pair[0])

    return apply_and_swap


# ======================================================================================


class EconometricSemanticParser(Domain):
    def __init__(self, db):
        self.db = db
        self.column_names = db.get_column_names()

    def train_examples(self):
        return train_examples

    def test_examples(self):
        return [
            Example(
                input="Find the mean of ENTERQ",
                semantics=("findMean", ("get_col", "ENTERQ")),
                to_lower=True,
                denotation=2 / 3,
            ),
        ]

    def dev_examples(self):
        return []

    def rules(self):
        optional_words = [
            "the",
            "what",
            "what's",
            "whats",
            "find",
            "show",
            "is",
            "in",
            "of",
            "how",
            "many",
            "are",
            "which",
            "that",
            "with",
            "has",
            "major",
            "does",
            "have",
            "where",
            "me",
            "there",
            "give",
            "name",
            "all",
            "a",
            "by",
            "you",
            "to",
            "tell",
            "other",
            "it",
            "do",
            "whose",
            "show",
            "one",
            "on",
            "for",
            "can",
            "whats",
            "urban",
            "them",
            "list",
            "exist",
            "each",
            "could",
            "about",
            ".",
            "?",
        ]

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

        rules_arguments = [Rule("$Arguments", "$Column", sems_0x)]

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

    def annotators(self):
        syns_to_names = [
            tuple(func)
            for func in zip(
                self.db.get_function_synonyms(), self.db.get_function_names()
            )
        ]
        return [FunctionAnnotator(syns_to_names), ColumnAnnotator(self.column_names)]

    def empty_denotation_feature(self, parse):
        features = defaultdict(float)
        if parse.denotation == ():
            features["empty_denotation"] += 1.0
        return features

    def features(self, parse):
        features = defaultdict(float)
        # TODO: turning off rule features seems to screw up learning
        # figure out what's going on here
        # maybe make an exercise of it!
        # Actually it doesn't seem to mess up final result.
        # But the train accuracy reported during SGD is misleading?
        features.update(rule_features(parse))
        features.update(self.empty_denotation_feature(parse))
        # EXERCISE: Experiment with additional features.
        return features

    def weights(self):
        weights = defaultdict(float)
        weights["empty_denotation"] = -1.0
        return weights

    def grammar(self):
        return Grammar(rules=self.rules(), annotators=self.annotators())

    def get_ops(self):

        return {
            "~": lambda x: -x,
            "+": lambda x, y: x + y,
            "-": lambda x, y: x - y,
            "/": lambda x, y: x / y,
            "*": lambda x, y: x * y,
            "avg": lambda x, y: (x + y) / 2,
            "^2": lambda x: x ** 2,
            "^3": lambda x: x ** 3,
            "^1/2": lambda x: x ** (1 / 2),
            """'findMean': lambda x: econlib.findMean(x),
            'findStd': lambda x: econlib.findStd(x),
            'findVar': lambda x: econlib.findVar(x),
            'findMax': lambda x: econlib.findMax(x),
            'findMin': lambda x: econlib.findMin(x),"""
            "getCol": lambda col: self.df[col],
            "applyFunctionManyTimes": lambda f, args: [f(arg) for arg in args],
            "getAllColumns": lambda x: [self.df[col] for col in self.column_names],
        }

    def execute(self, semantics):
        """
        print(self)
        print(semantics)
        print()
        """
        if isinstance(semantics, tuple):
            if semantics[0]:
                # op = self.ops[semantics[0]]
                args = [self.execute(arg) for arg in semantics[1:]]
                if semantics[0] == "getCol":
                    return self.db.sheets[0].df[
                        args[0]
                    ]  # TO-DO: FIX THIS SHIT, WE CANT HAVE GET COL FUNCTIONING LIKE THIS. WE SHOULD HAVE NO GET COL AT ALL.
                else:
                    return self.db.execute_func(semantics[0], args)
        else:
            return semantics

    def metrics(self):
        return denotation_match_metrics()

    def training_metric(self):
        return DenotationAccuracyMetric()


# demos and experiments ================================================================

if __name__ == "__main__":
    df = train_example_matrix()
    databook = DataBook()
    databook.add_df(df)
    domain = EconometricSemanticParser(databook)
    evaluate_for_domain(domain, print_examples=True)
    # evaluate_dev_examples_for_domain(domain)
    # train_test_for_domain(domain, seed=1)
    # test_executor(domain)
    # sample_wins_and_losses(domain, metric=DenotationOracleAccuracyMetric())
    # learn_lexical_semantics(domain, seed=1)
    interact(domain, "the largest city in the largest state", T=0)
    # find_best_rules(domain)
