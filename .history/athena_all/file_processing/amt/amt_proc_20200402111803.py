import csv
import sys

from athena_all.sem_parser.grammar import Example, handle_punctuation


class AMT_example:
    ### Amazon Mechanichal Turk examples
    def __init__(self, original_sentence, rewritten_sentence):
        self.original_sentence = original_sentence
        self.rewritten_sentence = rewritten_sentence


def create_examples_from_amt(
    denoted_examples, filename="./file_processing/amt/amt_data/amt.csv"
):
    amt_exs = []

    # Read in the given file for processing to create training examples.
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == "Approved":
                amt_exs.append(AMT_example(row[1], row[2]))

    examples_formatted = []

    # Go through all amazon mechanchial turk
    for amt_ex in amt_exs:
        ex_found = False

        # iterate through examples in our training set
        for denoted_ex in denoted_examples:

            # if the source matches than use the training examples semantics
            # and denotation for creating the new training example
            if (
                handle_punctuation(amt_ex.original_sentence.lower())
                == denoted_ex.input.lower()
            ) and not ex_found:
                examples_formatted.append(
                    Example(
                        amt_ex.rewritten_sentence,
                        denoted_ex.parse,
                        denoted_ex.semantics,
                        denoted_ex.denotation,
                        to_lower=True,
                    )
                )
                ex_found = True
        if not ex_found:
            print(f"couldnt find::: {amt_ex.original_sentence.lower()}")

    print(
        f"AMT processing: number of overall amazon mechanichal turk examples: {len(amt_exs)}"
    )
    print(
        f"AMT processing: number of examples succesfully parsed for training {len(examples_formatted)}"
    )

    return examples_formatted


if __name__ == "__main__":
    if sys.argv[1] is not None:
        create_examples_from_amt(sys.argv[1])
