from collections import defaultdict, Iterable
from itertools import product
from types import FunctionType

from athena_all.sem_parser.grammar._rule import Rule, is_cat, is_optional
from athena_all.sem_parser.grammar._parse import Parse, apply_semantics

DEBUG = False
MAX_CELL_CAPACITY = 1000  # upper bound on number of parses in one chart cell

# Grammar ======================================================================


class Grammar:
    def __init__(self, rules=[], annotators=[], start_symbol="$ROOT"):
        self.categories = set()
        self.lexical_rules = defaultdict(list)
        self.unary_rules = defaultdict(list)
        self.binary_rules = defaultdict(list)
        self.annotators = annotators
        self.start_symbol = start_symbol
        for rule in rules:
            self.add_rule(rule)
        print("Created grammar with %d rules" % len(rules))

    def parse_input(self, input):
        """
        Returns the list of parses for the given input which can be derived
        using this grammar.
        """
        return parse_input(self, input)

    def add_rule(self, rule):
        if rule.contains_optionals():
            self.add_rule_containing_optional(rule)
        elif rule.is_lexical():
            self.lexical_rules[rule.rhs].append(rule)
        elif rule.is_unary():
            self.unary_rules[rule.rhs].append(rule)
        elif rule.is_binary():
            self.binary_rules[rule.rhs].append(rule)
        elif all([is_cat(rhsi) for rhsi in rule.rhs]):
            self.add_n_ary_rule(rule)
        else:
            self.add_mixed_rule(rule)
            # EXERCISE: handle this case.
            raise Exception("RHS mixes terminals and non-terminals: %s" % rule)

    def add_rule_containing_optional(self, rule):
        """
        Handles adding a rule which contains an optional element on the RHS.
        We find the leftmost optional element on the RHS, and then generate
        two variants of the rule: one in which that element is required, and
        one in which it is removed.  We add these variants in place of the
        original rule.  (If there are more optional elements further to the
        right, we'll wind up recursing.)

        For example, if the original rule is:

            Rule('$Z', '$A ?$B ?$C $D')

        then we add these rules instead:

            Rule('$Z', '$A $B ?$C $D')
            Rule('$Z', '$A ?$C $D')
        """
        # Find index of the first optional element on the RHS.
        first = next((idx for idx, elt in enumerate(rule.rhs) if is_optional(elt)), -1)
        assert first >= 0
        assert len(rule.rhs) > 1, "Entire RHS is optional: %s" % rule
        prefix = rule.rhs[:first]
        suffix = rule.rhs[(first + 1) :]
        # First variant: the first optional element gets deoptionalized.
        deoptionalized = (rule.rhs[first][1:],)
        self.add_rule(Rule(rule.lhs, prefix + deoptionalized + suffix, rule.sem))
        # Second variant: the first optional element gets removed.
        # If the semantics is a value, just keep it as is.
        sem = rule.sem
        # But if it's a function, we need to supply a dummy argument for the removed element.
        if isinstance(rule.sem, FunctionType):
            sem = lambda sems: rule.sem(sems[:first] + [None] + sems[first:])
        self.add_rule(Rule(rule.lhs, prefix + suffix, sem))

    def add_n_ary_rule(self, rule):
        """
        Handles adding a rule with three or more non-terminals on the RHS.
        We introduce a new category which covers all elements on the RHS except
        the first, and then generate two variants of the rule: one which
        consumes those elements to produce the new category, and another which
        combines the new category which the first element to produce the
        original LHS category.  We add these variants in place of the
        original rule.  (If the new rules still contain more than two elements
        on the RHS, we'll wind up recursing.)

        For example, if the original rule is:

            Rule('$Z', '$A $B $C $D')

        then we create a new category '$Z_$A' (roughly, "$Z missing $A to the left"),
        and add these rules instead:

            Rule('$Z_$A', '$B $C $D')
            Rule('$Z', '$A $Z_$A')
        """

        def add_category(base_name):
            assert is_cat(base_name)
            name = base_name
            while name in self.categories:
                name = name + "_"
            self.categories.add(name)
            return name

        category = add_category("%s_%s" % (rule.lhs, rule.rhs[0]))
        self.add_rule(Rule(category, rule.rhs[1:], lambda sems: sems))
        self.add_rule(
            Rule(
                rule.lhs,
                (rule.rhs[0], category),
                lambda sems: apply_semantics(rule, [sems[0]] + sems[1]),
            )
        )

    def add_mixed_rule(self, rule):
        """
        Handles rules that mix terminal and non-terminal symbol on the right hand side.
        """
        assert any([not is_cat(rhsi) for rhsi in rule.rhsi])

    def parse_input(self, input):
        """
        Returns the list of parses for the given input which can be derived using
        the given grammar.
        """
        tokens = input.split()
        # TODO: populate chart with tokens?  that way everything is in the chart
        chart = defaultdict(list)
        for j in range(1, len(tokens) + 1):
            for i in range(j - 1, -1, -1):
                self.apply_annotators(chart, tokens, i, j)
                self.apply_lexical_rules(chart, tokens, i, j)
                self.apply_binary_rules(chart, i, j)
                self.apply_unary_rules(chart, i, j)

        parses = chart[(0, len(tokens))]

        if DEBUG:
            print_chart(chart)
            print(
                "\n".join(
                    [
                        f"\nsemantics: {str(parse.semantics)}  \n parse:  {str(parse)}"
                        for parse in parses
                    ]
                )
            )

        if self.start_symbol:
            parses = [parse for parse in parses if parse.rule.lhs == self.start_symbol]

        return parses

    def apply_annotators(self, chart, tokens, i, j):
        """Add parses to chart cell (i, j) by applying annotators."""
        if hasattr(self, "annotators"):
            for annotator in self.annotators:
                for category, semantics in annotator.annotate(tokens[i:j]):
                    if not check_capacity(chart, i, j):
                        return
                    rule = Rule(category, tuple(tokens[i:j]), semantics)
                    chart[(i, j)].append(Parse(rule, tokens[i:j]))

    def apply_lexical_rules(self, chart, tokens, i, j):
        """Add parses to chart cell (i, j) by applying lexical rules."""
        for rule in self.lexical_rules[tuple(tokens[i:j])]:
            if not check_capacity(chart, i, j):
                return
            chart[(i, j)].append(Parse(rule, tokens[i:j]))

    def apply_binary_rules(self, chart, i, j):
        """Add parses to chart cell (i, j) by applying binary rules."""
        for k in range(i + 1, j):
            for parse_1, parse_2 in product(chart[(i, k)], chart[(k, j)]):
                for rule in self.binary_rules[(parse_1.rule.lhs, parse_2.rule.lhs)]:
                    if not check_capacity(chart, i, j):
                        return
                    chart[(i, j)].append(Parse(rule, [parse_1, parse_2]))

    def apply_unary_rules(self, chart, i, j):
        """Add parses to chart cell (i, j) by applying unary rules."""
        # Note that the last line of this method can add new parses to chart[(i,
        # j)], the list over which we are iterating.  Because of this, we
        # essentially get unary closure "for free".  (However, if the grammar
        # contains unary cycles, we'll get stuck in a loop, which is one reason for
        # check_capacity().)
        for parse in chart[(i, j)]:
            for rule in self.unary_rules[(parse.rule.lhs,)]:
                if not check_capacity(chart, i, j):
                    return
                chart[(i, j)].append(Parse(rule, [parse]))

    def print_grammar(self):
        def all_rules(rule_index):
            return [rule for rules in list(rule_index.values()) for rule in rules]

        def print_rules_sorted(rules):
            for s in sorted([str(rule) for rule in rules]):
                print("  " + s)

        print("Lexical rules:")
        print_rules_sorted(all_rules(self.lexical_rules))
        print("Unary rules:")
        print_rules_sorted(all_rules(self.unary_rules))
        print("Binary rules:")
        print_rules_sorted(all_rules(self.binary_rules))


# Important for catching e.g. unary cycles.
max_cell_capacity_hits = 0


def check_capacity(chart, i, j):
    global max_cell_capacity_hits
    if len(chart[(i, j)]) >= MAX_CELL_CAPACITY:
        # print 'Cell (%d, %d) has reached capacity %d' % (
        #     i, j, MAX_CELL_CAPACITY)
        max_cell_capacity_hits += 1
        lg_max_cell_capacity_hits = math.log(max_cell_capacity_hits, 2)
        if int(lg_max_cell_capacity_hits) == lg_max_cell_capacity_hits:
            print(
                "Max cell capacity %d has been hit %d times"
                % (MAX_CELL_CAPACITY, max_cell_capacity_hits)
            )
        return False
    return True


def print_chart(chart):
    """Print the chart.  Useful for debugging."""
    spans = sorted(list(chart.keys()), key=(lambda span: span[0]))
    spans = sorted(spans, key=(lambda span: span[1] - span[0]))
    for span in spans:
        if len(chart[span]) > 0:
            print("%-12s" % str(span), end=" ")
            print(chart[span][0])
            for entry in chart[span][1:]:
                print("%-12s" % " ", entry)

