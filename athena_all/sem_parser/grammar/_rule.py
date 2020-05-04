# Rule =========================================================================


class Rule:
    """Represents a CFG rule with a semantic attachment."""

    def __init__(
        self, lhs, rhs, sem=None, add_all_permutations=False, add_optionals_btw=False
    ):
        self.lhs = lhs
        self.rhs = tuple(rhs.split()) if isinstance(rhs, str) else rhs
        self.sem = sem
        self.validate_rule()
        self.add_all_permutations = add_all_permutations
        self.add_optionals_btw = add_optionals_btw

    def __eq__(self, other):
        return (self.lhs, self.rhs) == (other.lhs, other.rhs)

    def __hash__(self):
        return hash(self.lhs, self.rhs)

    def __str__(self):
        """Returns a string representation of this Rule."""
        return "Rule" + str((self.lhs, " ".join(self.rhs), self.sem))

    def validate_rule(self):
        """Returns true iff the given Rule is well-formed."""
        assert is_cat(self.lhs), "Not a category: %s" % self.lhs
        assert isinstance(self.rhs, tuple), "Not a tuple: %s" % self.rhs
        for rhs_i in self.rhs:
            assert isinstance(rhs_i, str), "Not a string: %s" % rhs_i

    def is_unary(self):
        """
        Returns true iff the given Rule is a unary compositional rule, i.e.,
        contains only a single category (non-terminal) on the RHS.
        """
        return len(self.rhs) == 1 and is_cat(self.rhs[0])

    def is_lexical(self):
        """
        Returns true iff the given Rule is a lexical rule, i.e., contains only
        words (terminals) on the RHS.
        """
        return all([not is_cat(rhsi) for rhsi in self.rhs])

    def is_binary(self):
        """
        Returns true iff the given Rule is a binary compositional rule, i.e.,
        contains exactly two categories (non-terminals) on the RHS.
        """
        return len(self.rhs) == 2 and is_cat(self.rhs[0]) and is_cat(self.rhs[1])

    def contains_optionals(self):
        """Returns true iff the given Rule contains any optional items on the RHS."""
        return any([is_optional(rhsi) for rhsi in self.rhs])


""" =========================================================================================
    Static Methods useful for evaluating labels for rules. Used both in 
    Rule datatype and in the parse datatype.
============================================================================================= """


def is_cat(label):
    """
    Returns true iff the given label is a CATEGORY (non-terminal), i.e., is
    marked with an initial '$'.
    """
    return label.startswith("$")


def is_optional(label):
    """
    Returns true iff the given RHS item is optional, i.e., is marked with an
    initial '?'.
    """
    return label.startswith("?") and len(label) > 1

