import re

def tokenize(text):
    text = text.lower()

    tokens = re.findall(r"[a-z0-9]+", text)

    return tokens