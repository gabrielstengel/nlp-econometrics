
from athena_all.file_processing.excel import DataReader
from athena_all.sem_parser.parser_model import EconometricSemanticParser
from athena_all.databook.databook import DataBook
from athena_all.sem_parser.grammar._learning import latent_sgd

import readline
from athena_all.sem_parser.grammar._example import Example
from athena_all.sem_parser.experiment import print_parses


class Athena():

    def __init__(self, ):
        self.datareader = None
        self.queries = []
        self.databook = None


    def _process_first_query(self, query):
        try:
            dr = DataReader(query.lower())
        except:
            return QueryResponse("Filename does not exist. Please enter a different file or help for more information.",
                                 result=None,
                                 computation_time=0,
                                 flags=None)
                                 

        self.datareader = dr
        self.dfs = dr.get_all_sheets()
        sheet_heads = '\n'.join([str(df.head()) for df in self.dfs])
        
        self.databook = DataBook()
        self.databook.add_dfs(self.dfs)

        ### Only works with data with one sheet right now
        self.domain = EconometricSemanticParser(self.databook)
        self.model = self.domain.model()
        self.model = latent_sgd(model=self.model,
                       examples=self.domain.train_examples(),
                       training_metric=self.domain.training_metric(),
                       T=10)
        utter = "Succesfully loaded the dataset.\n"
        return QueryResponse(
            response = utter + f"{query} contained {len(self.dfs)} sheets." if len(self.dfs) > 1 else utter + f"{query} contained only one sheet.",
            result = None,
            computation_time=0,
            flags=['Unkown']
        )


    def process_query(self, query):

        # if datareader hasn't been initialized process this as a first query.
        if not self.datareader:
            qresult = self._process_first_query(query)

        # Process the query normally on the data.
        else:
            qresult = self._process_query(query.lower())

        self.queries.append(qresult)
        return qresult.response, qresult.result

    def _process_query(self, query, print_debugging = False):
        example = Example(input=query, to_lower=True)
        parses = self.model.parse_input(example.input)
        if parses:
            if print_debugging:
                print_parses(example, parses)

            return QueryResponse(
                response = parses[0].utterance,
                result = parses[0].denotation,
                computation_time=0,
                flags=['Unkown']
            )   
            
        return QueryResponse(
                response = f"Sorry, I didn't understand that.",
                result = None,
                computation_time=0,
                flags=['Unknown', 'UTP'] # UTP means "Unable to Parse"
            )    


class QueryResponse(object):

    def __init__(self, response, result=None, computation_time=None, flags=None):
        self.response = response
        self.result = result
        self.computation_time = computation_time
        self.flags = flags

        self._process_flags()

    def _process_flags(self):
        if self.flags:
            if "Correct" in self.flags:
                self.accuracy = 1
            elif "Unknown" in self.flags:
                self.accuracy = -1
            elif "Incorrect" in self.flags:
                self.accuracy = 0
        else:
            ## No definition of accuracy for this query
            self.accuracy = -1

def main():
    client = Athena()
    query = input("Hi! Please enter the name of the file you'd like to work with.\n>>> ")
    resp = client.process_query(query)
    print(resp[0])
    print()
    while True:
        try:
            query = input(">>> ")
        except EOFError:
            print("\nBye!")
            return
        print()
        print(client.process_query(query)[0])
        print()
    


        
    
    

if __name__ == "__main__":
    main()