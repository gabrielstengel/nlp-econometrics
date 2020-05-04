class QueryResult:
    def __init__(self, denotation, nl_utterance):
        self.denotation = denotation
        self.utterance = nl_utterance

    def get_denotation(self):
        return self.denotation

    def get_utterance(self):
        return self.utterance
