punctuationList = [
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    # "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    # "<",
    # "=",
    # ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    # "_",
    "`",
    "{",
    "|",
    "}",
    "~",
]


def handle_punctuation(utterance):
    return utterance.translate(
        str.maketrans({key: " {0} ".format(key) for key in punctuationList})
    )
