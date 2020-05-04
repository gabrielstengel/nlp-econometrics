from collections import defaultdict
from numbers import Number
import string

import pandas as pd

from athena_all.sem_parser.traindata.econ_train_examples2 import (
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
    standard_metrics,
)

from athena_all.sem_parser._rules import generate_rules
from athena_all.file_processing import create_examples_from_amt


from athena_all.databook import DataBook

# EconometricSemanticParser ============================================================
# ======================================================================================

EPSILON = 0.001


class EconometricSemanticParser(Domain):
    """ The object that will parse inputs. Takes train examples, test examples."""

    def __init__(self, db):
        self.db = db
        self.column_names = db.get_column_names()

    def train_examples(self):
        return train_examples[:-1]  # + create_examples_from_amt()

    def test_examples(self):
        return train_examples[-1:]

    def _process_examples(self, examples):
        text = text.translate(
            str.maketrans({key: " {0} ".format(key) for key in string.punctuation})
        )

    def dev_examples(self):
        return []

    def rules(self):
        return generate_rules(self.db)

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

    def execute(self, semantics):
        """ Method for executing the logical statement.  Recursively calls function 
        on its arguments. """

        if isinstance(semantics, tuple):
            if semantics[0]:
                args = [self.execute(arg) for arg in semantics[1:]]
                return self.db.execute_func(semantics[0], args)
        else:
            return semantics

    def metrics(self):
        return standard_metrics(EPSILON)

    def training_metric(self):
        return DenotationAccuracyMetric(EPSILON)


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
