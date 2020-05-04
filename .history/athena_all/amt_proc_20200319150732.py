import csv
import sys
import athena_all.econ_train_examples as texamps

class AMT_example():
    def __init__(self, req, utter):
        self.req = req
        self.utter = utter

def main(filename):
    examples = []
    print("in main")
    
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'Approved':
                examples.append(AMT_example(row[1], row[2]))
               
    real_examples =  texamps.train_examples
    for ex in examples:
        for real_ex in examples:
            if ex.req.lower() == real_ex.input()
    
    
    print(len(examples))



if __name__ == '__main__':
    if sys.argv[1] is not None:
        main(sys.argv[1])