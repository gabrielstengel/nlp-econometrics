
import Levenshtein as lev
import numpy as np

from athena_all.econlib_old import *

# ColumnAnnotator =============================================================

class Annotator:
    """A base class for annotators."""
    def annotate(self, tokens):
        """Returns a list of pairs, each a category and a semantic representation."""
        return []

# TokenAnnotator =============================================================

class TokenAnnotator(Annotator):
    def annotate(self, tokens):
        if len(tokens) == 1:
            return [('$Token', tokens[0])]
        else:
            return []

# ColumnAnnotator =============================================================

def get_function(query, mapping=fmaps):
    ''' Given a string return the function whose name is most similair based on levenstien distance. 
        Also returns the name of the function for verification. '''

    func_sim = np.argmax([np.max(np.array([lev.ratio(term, query) for term in fmap[0]])) for fmap in fmaps])
    return (fmaps[func_sim][0][0], fmaps[func_sim][1])


# FunctionAnnotator =============================================================

class FunctionAnnotator(Annotator):
    def __init__(self, functionMaps):
        assert len(functionMaps[0]) == 2, "Function map should only contain synonyms and semantics."
        assert type(functionMaps[0][0]) is list, "Function Synonyms must be a list."
        self.fmaps = functionMaps
    
    def annotate(self, tokens):
        phrase = ' '.join(tokens)
        for funcs in self.fmaps:
            if phrase in funcs[0]:
                return [('$'+funcs[1], funcs[1])]
        return []

# NumberAnnotator =============================================================

class NumberAnnotator(Annotator):
    def annotate(self, tokens):
        if len(tokens) == 1:
            try:
                value = float(tokens[0])
                if int(value) == value:
                    value = int(value)
                return [('$Number', value)]
            except ValueError:
                pass
        return []
    

# ColumnAnnotator =============================================================

# EXERCISE: Make it more robust, using string edit distance or minhashing.
class ColumnAnnotator(Annotator):
    def __init__(self, column_list):
        self.column_list = [col.lower() for col in column_list]

    def annotate(self, tokens):
        if len(tokens) == 1:
            if tokens[0] in self.column_list:
                return [('$Column', tokens[0])]
            elif tokens[0].endswith("'s"):
                if len(tokens[0][:-2]) > 0 and tokens[0][:-2] in self.column_list:
                    print(f'returning {[("$Column", tokens[0][:-2])]}')
                    return [('$Column', tokens[0][:-2])]
            elif tokens[0].endswith("s"):
                if len(tokens[0][:-1]) > 0 and tokens[0][:-1] in self.column_list:
                    return [('$Column', tokens[0][:-1])]
        return []


if __name__ == '__main__':
    annotators = [TokenAnnotator(), NumberAnnotator()]
    tokens = 'four score and 30 years ago'.split()
    for j in range(1, len(tokens) + 1):
        for i in range(j - 1, -1, -1):
            annotations = [a for anno in annotators for a in anno.annotate(tokens[i:j])]
            print('(%d, %d): %s => %s' % (i, j, ' '.join(tokens[i:j]), annotations))
