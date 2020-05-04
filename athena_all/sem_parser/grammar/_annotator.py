import Levenshtein as lev
from word2number import w2n
import numpy as np

# Basic Annotator type =============================================================


class Annotator:
    """A base class for annotators."""

    def annotate(self, tokens):
        """Returns a list of pairs, each a category and a semantic representation."""
        return []


# TokenAnnotator =============================================================


class TokenAnnotator(Annotator):
    def annotate(self, tokens):
        if len(tokens) == 1:
            return [("$Optional", tokens[0])]
        else:
            return []


# FunctionAnnotator =============================================================


class FunctionAnnotator(Annotator):
    def __init__(self, syns_to_names):
        assert (
            len(syns_to_names[0]) == 2
        ), "Function map should only contain synonyms and semantics."
        assert type(syns_to_names[0][0]) is list, "Function Synonyms must be a list."
        self.syns_to_names = syns_to_names

    def get_similair_function(self, query):
        """ Given a string return the function whose name is most similair based on levenstien distance. 
        Also returns the name of the function for verification. """

        func_sim = np.argmax(
            [
                np.max(np.array([lev.ratio(term, query) for term in fmap[0]]))
                for fmap in self.syns_to_names
            ]
        )
        return (self.syns_to_names[func_sim][0][0], self.syns_to_names[func_sim][1])

    def annotate(self, tokens):
        annotations = []

        phrase = " ".join(tokens)
        for funcs in self.syns_to_names:
            if phrase in funcs[0]:
                annotations += [("$" + funcs[1], funcs[1])]
        return annotations


# NumberAnnotator =============================================================


class NumberAnnotator(Annotator):
    def annotate(self, tokens):
        if len(tokens) == 1:
            try:
                value = w2n.word_to_num(tokens[0])
                return [("$Number", value)]
            except ValueError:
                pass
        else:
            if "." in tokens or "," in tokens:
                phrase = "".join(tokens).replace(",", "")
                try:
                    value = float(phrase)
                    return [("$Number", value)]
                except ValueError:
                    pass
            else:
                phrase = " ".join(tokens)
                try:
                    value = w2n.word_to_num(phrase)
                    return [("$Number", value)]
                except:
                    pass
        return []


# ColumnAnnotator =============================================================

# EXERCISE: Make it more robust, using string edit distance or minhashing.
class ColumnAnnotator(Annotator):
    def __init__(self, column_list):
        self.column_list = [col.lower() for col in column_list]

    def annotate(self, tokens, threshhold = 0.7):
        if len(tokens) == 1:
            if tokens[0] in self.column_list:
                return [("$Column", tokens[0])]
            elif tokens[0].endswith("'s"):
                if len(tokens[0][:-2]) > 0 and tokens[0][:-2] in self.column_list:
                    # print(f'returning {[("$Column", tokens[0][:-2])]}')
                    return [("$Column", tokens[0][:-2])]
            elif tokens[0].endswith("s"):
                if len(tokens[0][:-1]) > 0 and tokens[0][:-1] in self.column_list:
                    return [("$Column", tokens[0][:-1])]
            
            closest = 0.0
            closest_col = None
            for col in self.column_list:
                how_close = lev.ratio(tokens[0], col) 
                if how_close >= closest:
                    closest = how_close
                    closest_col = col
            if closest > threshhold:
                return [("$Column", closest_col)]
        
        else:
            tokens = "".join(tokens)
            closest = 0.0
            closest_col = None
            for col in self.column_list:
                how_close = lev.ratio(tokens, col) 
                if how_close >= closest:
                    closest = how_close
                    closest_col = col
            if closest > threshhold+.1:
                return [("$Column", closest_col)]

                
        return []


# MainFunction for Testing =============================================================

if __name__ == "__main__":
    annotators = [TokenAnnotator(), NumberAnnotator()]
    tokens = "four score and 30 years ago".split()
    for j in range(1, len(tokens) + 1):
        for i in range(j - 1, -1, -1):
            annotations = [a for anno in annotators for a in anno.annotate(tokens[i:j])]
            print("(%d, %d): %s => %s" % (i, j, " ".join(tokens[i:j]), annotations))
