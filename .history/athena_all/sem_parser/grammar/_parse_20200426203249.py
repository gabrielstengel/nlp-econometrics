from collections import Iterable
from six import StringIO
from types import FunctionType

from athena_all.sem_parser.grammar._rule import Rule, is_cat, is_optional


# Parse ========================================================================


class Parse:
    """ 
    Simple container class. Stores the rule used to build the parse and  the children
    to which the Rule is applied. If the rule was a lexical rule then the children
    are just tokens, otherwise the children are other Parse objects.                
    """

    def __init__(self, rule, children):
        self.rule = rule
        self.children = tuple(children[:])
        self.semantics = self.compute_semantics()
        self.score = float("NaN")
        self.denotation = None
        self.utterance = None
        self.validate_parse()

    def __str__(self):
        child_strings = [str(child) for child in self.children]
        return "(%s %s)" % (self.rule.lhs, " ".join(child_strings))

    def validate_parse(self):
        assert isinstance(self.rule, Rule), "Not a Rule: %s" % self.rule
        assert isinstance(self.children, Iterable)
        assert len(self.children) == len(self.rule.rhs)
        for i in range(len(self.rule.rhs)):
            if is_cat(self.rule.rhs[i]):
                assert self.rule.rhs[i] == self.children[i].rule.lhs
            else:
                assert self.rule.rhs[i] == self.children[i]

    def compute_semantics(self):
        if self.rule.is_lexical():
            return self.rule.sem
        else:
            child_semantics = [child.semantics for child in self.children]
            return apply_semantics(self.rule, child_semantics)

    def parse_to_pretty_string(self, indent=0, show_sem=False):
        def indent_string(level):
            return "  " * level

        def label(parse):
            if show_sem:
                return "(%s %s)" % (parse.rule.lhs, parse.semantics)
            else:
                return parse.rule.lhs

        def to_oneline_string(parse):
            if isinstance(parse, Parse):
                child_strings = [to_oneline_string(child) for child in parse.children]
                return "[%s %s]" % (label(parse), " ".join(child_strings))
            else:
                return str(parse)

        def helper(parse, level, output):
            line = indent_string(level) + to_oneline_string(parse)
            if len(line) <= 100:
                print(line, file=output)
            elif isinstance(parse, Parse):
                print(indent_string(level) + "[" + label(parse), file=output)
                for child in parse.children:
                    helper(child, level + 1, output)
                # TODO: Put closing parens to end of previous line, not dangling alone.
                print(indent_string(level) + "]", file=output)
            else:
                print(indent_string(level) + parse, file=output)

        output = StringIO()
        helper(self, indent, output)
        return output.getvalue()[:-1]  # trim final newline


""" =========================================================================================
    Static Methods useful for evaluating the semantics according to a rule.
    Used here and in _grammar.
============================================================================================= """


def apply_semantics(rule, sems):
    # Note that this function would not be needed if we required that semantics
    # always be functions, never bare values.  That is, if instead of
    # Rule('$E', 'one', 1) we required Rule('$E', 'one', lambda sems: 1).
    # But that would be cumbersome.
    if isinstance(rule.sem, FunctionType):
        return rule.sem(sems)
    else:
        return rule.sem
