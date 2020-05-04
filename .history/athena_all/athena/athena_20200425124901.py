
from athena_all.file_processing.excel import DataReader
from athena_all.sem_parser.parser_model import EconometricSemanticParser

import readline
from athena_all.databook.databook import DataBook

class Athena():

    def __init__(self, ):
        self.datareader = None
        self.queries = []
        self.databook = None


    def _process_first_query(self, query):
        try:
            print("DEBUG: Trying to initialize dataloader")
            dr = DataReader(query.lower())
        except:
            print("Query: " + query)
            return QueryResponse("Filename does not exist. Please enter a different file or help for more information.",
                                 result=None,
                                 computation_time=0,
                                 flags=None)
                                 

        self.datareader = dr
        self.dfs = dr.get_all_sheets()
        sheet_heads = '\n'.join([str(df.head()) for df in self.dfs])
        
        databook = DataBook()
        databook.add_dfs()

        ### Only works with data with one sheet right now
        self.domain = EconometricSemanticParser(self.dfs[0])
        self.model = self.domain.model()
        self.model = latent_sgd(model=self.model,
                       examples=self.domain.train_examples(),
                       training_metric=self.domain.training_metric(),
                       T=10)

        return QueryResponse(
            response = f"{query} contained {len(self.dfs)} sheets." if len(self.dfs) > 1 else f"{query} contained only one sheet.",
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
        example = Example(input=query)
        parses = self.model.parse_input(query)
        if parses:
            if print_debugging:
                print_parses(example, parses)

            return QueryResponse(
                response = f"Possible answers are {[p.denotation for p in parses]}",
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


'''class QueryResponse(object):

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


'''