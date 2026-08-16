from src.index import InvertedIndex
from src.tokenizer import tokenize

index = InvertedIndex()

for i in range(1, 7):
    with open(f"data/{i}.txt", "r") as d:
        tokens = d.read()
    index.add_documents(i, tokenize(tokens))
    print(i, " ---> ", index.document_length(i))

test_keywords = ["internet", "energy", "brain", "knowledge", "protocol", "xyz"]

for i in test_keywords:
    print(i, " ---> ", index.search(i))