from athena_all.sem_parser.grammar import handle_punctuation


class Example:
    def __init__(
        self, input=None, parse=None, semantics=None, denotation=None, to_lower=True
    ):
        self.input = input.lower() if to_lower else input
        self.input = handle_punctuation(self.input)
        self.parse = parse
        self.semantics = semantics
        self.denotation = denotation

    def __str__(self):
        fields = []
        self.input != None and fields.append(
            "input='%s'" % self.input.replace("'", "'")
        )
        self.parse != None and fields.append("parse=%s" % self.parse)
        self.semantics != None and fields.append("semantics=%s" % str(self.semantics))
        self.denotation != None and fields.append(
            "denotation=%s" % str(self.denotation)
        )
        return "Example(%s)" % (", ".join(fields))


# Main Function for testing. ================================================================================

if __name__ == "__main__":
    examples = [
        Example(),
        Example(input="one plus one", semantics="(+ 1 1)", denotation=2),
        Example(input="two plus three", denotation=5),
        Example(input="utah", semantics="/state/utah", denotation=set(["/state/utah"])),
    ]
    for example in examples:
        print(example)
