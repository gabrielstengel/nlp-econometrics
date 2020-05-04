# Rule =========================================================================

class Rule:
    """Represents a CFG rule with a semantic attachment."""

    def __init__(self, lhs, rhs, sem=None):
        self.lhs = lhs
        self.rhs = tuple(rhs.split()) if isinstance(rhs, str) else rhs
        self.sem = sem
        validate_rule(self)

    def __str__(self):
        """Returns a string representation of this Rule."""
        return 'Rule' + str((self.lhs, ' '.join(self.rhs), self.sem))

def is_cat(label):
    """
    Returns true iff the given label is a category (non-terminal), i.e., is
    marked with an initial '$'.
    """
    return label.startswith('$')

def is_lexical(rule):
    """
    Returns true iff the given Rule is a lexical rule, i.e., contains only
    words (terminals) on the RHS.
    """
    return all([not is_cat(rhsi) for rhsi in rule.rhs])

def is_unary(rule):
    """
    Returns true iff the given Rule is a unary compositional rule, i.e.,
    contains only a single category (non-terminal) on the RHS.
    """
    return len(rule.rhs) == 1 and is_cat(rule.rhs[0])

def is_binary(rule):
    """
    Returns true iff the given Rule is a binary compositional rule, i.e.,
    contains exactly two categories (non-terminals) on the RHS.
    """
    return len(rule.rhs) == 2 and is_cat(rule.rhs[0]) and is_cat(rule.rhs[1])

def validate_rule(rule):
    """Returns true iff the given Rule is well-formed."""
    assert is_cat(rule.lhs), 'Not a category: %s' % rule.lhs
    assert isinstance(rule.rhs, tuple), 'Not a tuple: %s' % rule.rhs
    for rhs_i in rule.rhs:
        assert isinstance(rhs_i, str), 'Not a string: %s' % rhs_i

def is_optional(label):
    """
    Returns true iff the given RHS item is optional, i.e., is marked with an
    initial '?'.
    """
    return label.startswith('?') and len(label) > 1

def contains_optionals(rule):
    """Returns true iff the given Rule contains any optional items on the RHS."""
    return any([is_optional(rhsi) for rhsi in rule.rhs])


# Parse ========================================================================

class Parse:
    def __init__(self, rule, children):
        self.rule = rule
        self.children = tuple(children[:])
        self.semantics = compute_semantics(self)
        self.score = float('NaN')
        self.denotation = None
        validate_parse(self)

    def __str__(self):
        child_strings = [str(child) for child in self.children]
        return '(%s %s)' % (self.rule.lhs, ' '.join(child_strings))

def validate_parse(parse):
    assert isinstance(parse.rule, Rule), 'Not a Rule: %s' % parse.rule
    assert isinstance(parse.children, Iterable)
    assert len(parse.children) == len(parse.rule.rhs)
    for i in range(len(parse.rule.rhs)):
        if is_cat(parse.rule.rhs[i]):
            assert parse.rule.rhs[i] == parse.children[i].rule.lhs
        else:
            assert parse.rule.rhs[i] == parse.children[i]

def apply_semantics(rule, sems):
    # Note that this function would not be needed if we required that semantics
    # always be functions, never bare values.  That is, if instead of
    # Rule('$E', 'one', 1) we required Rule('$E', 'one', lambda sems: 1).
    # But that would be cumbersome.
    if isinstance(rule.sem, FunctionType):
        return rule.sem(sems)
    else:
        return rule.sem

def compute_semantics(parse):
    if is_lexical(parse.rule):
        return parse.rule.sem
    else:
        child_semantics = [child.semantics for child in parse.children]
        return apply_semantics(parse.rule, child_semantics)

def parse_to_pretty_string(parse, indent=0, show_sem=False):
    def indent_string(level):
        return '  ' * level
    def label(parse):
        if show_sem:
            return '(%s %s)' % (parse.rule.lhs, parse.semantics)
        else:
            return parse.rule.lhs
    def to_oneline_string(parse):
        if isinstance(parse, Parse):
          child_strings = [to_oneline_string(child) for child in parse.children]
          return '[%s %s]' % (label(parse), ' '.join(child_strings))
        else:
            return str(parse)
    def helper(parse, level, output):
        line = indent_string(level) + to_oneline_string(parse)
        if len(line) <= 100:
            print(line, file=output)
        elif isinstance(parse, Parse):
            print(indent_string(level) + '[' + label(parse), file=output)
            for child in parse.children:
                helper(child, level + 1, output)
            # TODO: Put closing parens to end of previous line, not dangling alone.
            print(indent_string(level) + ']', file=output)
        else:
            print(indent_string(level) + parse, file=output)
    output = StringIO()
    helper(parse, indent, output)
    return output.getvalue()[:-1]  # trim final newline
