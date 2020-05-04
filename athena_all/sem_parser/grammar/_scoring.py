from collections import defaultdict

from athena_all.sem_parser.grammar import Parse

# TODO: annotations are generating rule features -- they shouldn't.
def rule_features(parse):
    """
    Returns a map from (string representations of) rules to how often they were
    used in the given parse.
    """

    def collect_rule_features(parse, features):
        feature = str(parse.rule)
        features[feature] += 1.0
        for child in parse.children:
            if isinstance(child, Parse):
                collect_rule_features(child, features)

    features = defaultdict(float)
    collect_rule_features(parse, features)
    return features


def score(parse=None, feature_fn=None, weights=None):
    """Returns the inner product of feature_fn(parse) and weights."""
    assert parse and feature_fn and weights != None
    return sum(
        weights[feature] * value for feature, value in list(feature_fn(parse).items())
    )


class Model:
    def __init__(
        self,
        grammar=None,
        feature_fn=lambda parse: defaultdict(float),
        weights=defaultdict(float),
        executor=None,
    ):
        assert grammar
        self.grammar = grammar
        self.feature_fn = feature_fn
        self.weights = weights
        self.executor = executor

    # TODO: Should this become a static function, to match style of parsing.py?
    def parse_input(self, input):
        parses = self.grammar.parse_input(input)
        for parse in parses:
            if self.executor:
                res = self.executor(parse.semantics)
                parse.denotation = res.denotation
                parse.utterance = res.utterance
            parse.score = score(parse, self.feature_fn, self.weights)
        return sorted(parses, key=lambda parse: parse.score, reverse=True)
