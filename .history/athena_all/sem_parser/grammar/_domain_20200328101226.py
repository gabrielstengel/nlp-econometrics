"""
Defines the Domain class, which represents an object of the semantic parsing.
"""

from collections import defaultdict

from athena_all.sem_parser.grammar._metrics import (
    standard_metrics,
    SemanticsAccuracyMetric,
)
from athena_all.sem_parser.grammar._scoring import Model


class Domain:
    def train_examples(self):
        """Returns a list of training Examples suitable for the domain."""
        return []

    def dev_examples(self):
        """Returns a list of development Examples suitable for the domain."""
        return []

    def test_examples(self):
        """Returns a list of test Examples suitable for the domain."""
        return []

    def rules(self):
        """Returns a list of Rules suitable for the domain."""
        return []

    def annotators(self):
        """Returns a list of Annotators suitable for the domain."""
        return []

    def grammar(self):
        raise Exception("grammar() method not implemented")

    def features(self, parse):
        """
        Takes a parse and returns a map from feature names to float values.
        """
        return defaultdict(float)

    def weights(self):
        return defaultdict(float)

    def execute(self, semantics):
        """
        Executes a semantic representation and returns a denotation.  Both
        semantic representations and the denotations can be pretty much any
        Python values: numbers, strings, tuples, lists, sets, trees, and so on.
        Each domain will define its own spaces of semantic representations and
        denotations.
        """
        return None

    def model(self):
        return Model(
            grammar=self.grammar(),
            feature_fn=self.features,
            weights=self.weights(),
            executor=self.execute,
        )

    def metrics(self):
        """Returns a list of Metrics which are appropriate for the domain."""
        return standard_metrics()

    def training_metric(self):
        """
        Returns the evaluation metric which should be used to supervise training
        for this domain.
        """
        return SemanticsAccuracyMetric()
