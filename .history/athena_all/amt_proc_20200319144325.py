import csv
import sys

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
    
    print(len(examples))
    import pdb; pdb.set_trace()



if __name__ == '__main__':
    if sys.argv[1] is not None:
        main(sys.argv[1])