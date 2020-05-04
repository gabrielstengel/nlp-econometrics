import csv



class AMT_example(self):
    def __init__(self, req, utter):
        self.req = req
        self.utter = utter

def __main__(file_name):
    
    examples = []
    
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)
        if row[0] == 'Approved':
            examples.append(AMT_example(row[1], row[2]))