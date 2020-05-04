from collections import defaultdict
from numbers import Number
import string
import sys

import pandas as pd
import click

from athena_all.sem_parser.traindata.econ_train_examples2 import trainingExamples
from athena_all.sem_parser.experiment import (
    evaluate_dev_examples_for_domain,
    evaluate_for_domain,
    generate,
    interact,
    learn_lexical_semantics,
    print_sample_outcomes,
    train_test,
    train_test_for_domain,
)

from athena_all.sem_parser.grammar import (
    Grammar,
    Rule,
    FunctionAnnotator,
    ColumnAnnotator,
    NumberAnnotator,
    Domain,
    Example,
    DenotationAccuracyMetric,
    denotation_match_metrics,
    rule_features,
    standard_metrics,
)

from athena_all.sem_parser._rules import generate_rules
from athena_all.sem_parser._rules_extended import generate_extended_rules
from athena_all.file_processing import create_examples_from_amt


from athena_all.databook import DataBook
from athena_all.sem_parser.grammar._metrics import SemanticsOracleAccuracyMetric

# EconometricSemanticParser ============================================================
# ======================================================================================

EPSILON = 0.001


class EconometricSemanticParser(Domain):
    """ The object that will parse inputs. Takes train examples, test examples."""

    def __init__(self, db, examples=None):
        self.db = db
        self.column_names = db.get_column_names()
        if examples:
            self.examples = examples
        else:
            self.examples = [
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
            ]

    def train_examples(self):
        return self.examples[:-1]  # + create_examples_from_amt()

    def test_examples(self):
        return self.examples[-1:]

    def dev_examples(self):
        return []

    def rules(self):
        return generate_rules(self.db) + generate_extended_rules()

    def annotators(self):
        syns_to_names = [
            tuple(func)
            for func in zip(
                self.db.get_function_synonyms(), self.db.get_function_names()
            )
        ]
        return [
            FunctionAnnotator(syns_to_names),
            ColumnAnnotator(self.column_names),
            NumberAnnotator(),
        ]

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
        weights["versus"] = 1.0
        return weights

    def grammar(self):
        return Grammar(
            rules=self.rules(), annotators=self.annotators(), start_symbol="$ROOT"
        )

    def execute(self, semantics):
        """ Method for executing the logical statement.  Recursively calls function 
        on its arguments. """

        if isinstance(semantics, tuple):
            if semantics[0]:
                args = [self.execute(arg) for arg in semantics[1:]]
                return self.db.execute_func(semantics[0], args, denotation_only=False)
        else:
            return semantics

    def metrics(self):
        return standard_metrics(EPSILON)

    def training_metric(self):
        return DenotationAccuracyMetric(EPSILON)


# demos and experiments ================================================================


@click.command()
@click.option(
    "--experiment_only",
    type=bool,
    default=False,
    help="True if you just want a prompt.",
)
@click.option(
    "-n",
    "--print_none",
    type=bool,
    default=False,
    help="True if you don't want any examples to be printed.",
)
@click.option(
    "-p",
    "--print_all_examples",
    type=bool,
    default=True,
    help="True if you want all examples to be printed with parses, false if you only want unparsed examples to be printed.",
)
@click.argument(
    "function_names", nargs=-1,
)
def main(experiment_only, print_none, print_all_examples, function_names):
    """ FUNCTION_NAMES: the names of the functions you would like included in the example set. Enter none to use all examples. """
    ex_data = trainingExamples()
    df = ex_data.train_example_matrix()
    databook = DataBook()
    databook.add_df(df)

    if function_names:
        examples = []
        for func in function_names:
            examples += ex_data.get_examples_by_func(func)
    else:
        examples = ex_data.train_examples

    domain = EconometricSemanticParser(databook, examples)
    if not experiment_only:
        evaluate_for_domain(
            domain, print_examples=not print_none, print_all=print_all_examples
        )
    # domain.grammar().print_grammar()
    interact(domain, "Enter any statistical or econometric query:", T=0)


if __name__ == "__main__":
    main()

    # evaluate_dev_examples_for_domain(domain)
    # train_test_for_domain(domain, seed=1)
    # test_executor(domain)
    # sample_wins_and_losses(domain, metric=DenotationOracleAccuracyMetric())
    # learn_lexical_semantics(domain, seed=1)

    # find_best_rules(domain)
