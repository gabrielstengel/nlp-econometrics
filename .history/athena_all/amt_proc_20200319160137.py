import csv
import sys

import athena_all.econ_train_examples as texamps
from athena_all.sem_parser.grammar.example import Example


class AMT_example():
    ### Amazon Mechanichal Turk examples
    def __init__(self, original_sentence, rewritten_sentence):
        self.original_sentence = original_sentence
        self.rewritten_sentence = rewritten_sentence

def main(filename):
    amt_exs = []
    
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'Approved':
                amt_exs.append(AMT_example(row[1], row[2]))
               
               
    examples_formatted = []
    denoted_examples =  texamps.train_examples
    
    # Go through all amazon mechanchial turk
    for amt_ex in amt_exs:
        ex_found = False
        
        # iterate through examples in our training set
        for denoted_ex in denoted_examples:
            
            # if the source matches than use the training examples semantics 
            # and denotation for creating the new training example
            if amt_ex.original_sentence.lower() == denoted_ex.input:
                examples_formatted.append(Example(amt_ex.rewritten_sentence, denoted_ex.parse, denoted_ex.semantics, denoted_ex.denotation, to_lower=True))
                ex_found = True
        if not ex_found:
            print(f'couldnt find::: {amt_ex.original_sentence}')
                

    print(f'number of overall amazon mechanichal turk examples: {len(amt_exs)}')
    print(f'number of examples succesfully parsed for training {len(examples_formatted)}')


if __name__ == '__main__':
    if sys.argv[1] is not None:
        main(sys.argv[1])